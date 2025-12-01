import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sentinela Agroclimática", layout="wide", page_icon="🛰️")

# --- AUTENTICAÇÃO GOOGLE EARTH ENGINE (GEE) ---
# O 'MEU_PROJETO_ID' deve ser o ID do projeto no Google Cloud Platform.
# Sem isso, a API do Earth Engine pode rejeitar a conexão em ambientes novos.
MEU_PROJETO_ID = "projetoaero"

try:
    # Tenta inicializar. Se falhar (ex: token expirado), força a reautenticação.
    if MEU_PROJETO_ID:
        ee.Initialize(project=MEU_PROJETO_ID)
    else:
        ee.Initialize()
except Exception:
    ee.Authenticate()
    if MEU_PROJETO_ID:
        ee.Initialize(project=MEU_PROJETO_ID)
    else:
        ee.Initialize()

# --- CONSTANTES E DADOS PADRÃO ---
# Coordenadas iniciais centralizadas em uma região agrícola de Sorriso-MT (Capital do Agronegócio)
LAT_INICIAL = -12.2340821
LON_INICIAL = -55.7533832

#Quão abaixo da linha de Alerta a linha de Atenção deve ficar.
ajuste_atencao = 0.8

# 'Gabarito' de produtividade histórica. 
# Essencial para treinar a IA: ela precisa saber quanto choveu e quanto produziu 
# no passado para aprender a prever o futuro.
DADOS_HISTORICOS_PADRAO = pd.DataFrame({
    'safra': ['2017/2018', '2018/2019', '2019/2020', '2020/2021', '2021/2022', '2022/2023'],
    'produtividade_kg_ha': [3450, 3290, 3600, 3231, 3700, 3680]
})

# --- GLOSSÁRIO DE VARIÁVEIS DO SATÉLITE ---
# NDVI (Normalized Difference Vegetation Index): Mede o vigor da planta (cor verde).
# NDMI (Normalized Difference Moisture Index): Mede o conteúdo de água na folha.
# LST (Land Surface Temperature): Temperatura da superfície (estresse térmico).
# precip_mm_day: Precipitação média diária (chuva).
# ssm (Surface Soil Moisture): Umidade do solo superficial (satélite SMAP).
# VV, VH: Polarizações do Radar (Sentinel-1). Medem a estrutura física da planta e rugosidade,
#         sendo essenciais quando há nuvens bloqueando os sensores ópticos.
VARIAVEIS = ['NDVI', 'NDMI', 'LST', 'precip_mm_day', 'ssm', 'VV', 'VH']

# --- FUNÇÕES AUXILIARES (PROCESSAMENTO DE IMAGEM) ---

def processar_radar_s1(imagem):
    """
    Aplica filtro de suavização (Speckle) no Radar Sentinel-1.
    O Radar naturalmente tem um ruído granulado ("sal e pimenta").
    O filtro 'focal_mean' (média móvel) com raio de 50m suaviza esse ruído 
    para termos um sinal mais limpo da biomassa da cultura.
    """
    vv = imagem.select('VV').focal_mean(50, 'circle', 'meters', 1)
    vh = imagem.select('VH').focal_mean(50, 'circle', 'meters', 1)
    return imagem.addBands(vv.rename('VV'), overwrite=True).addBands(vh.rename('VH'), overwrite=True)

def processar_smap(imagem):
    """Padroniza o nome da banda de umidade do solo do satélite SMAP."""
    return imagem.select('soil_moisture_am').rename('ssm')

# --- FUNÇÕES DE BACKEND (LÓGICA PESADA DE COLETA E CÁLCULO) ---

@st.cache_data(ttl=3600)
def coletar_dados_gee(coords_poligono, ano_inicio, ano_fim, mes_inicio, mes_fim, passo_dias, escala):
    """
    Coleta a série temporal completa de todos os satélites para a região desenhada.
    Usa 'st.cache_data' para armazenar o resultado por 1 hora, evitando 
    requisições repetidas e lentas ao Google Earth Engine.
    """
    geometria = ee.Geometry.Polygon(coords_poligono)
    
    # Define o intervalo total de datas para busca no servidor
    data_inicio_str = f"{ano_inicio}-{mes_inicio:02d}-01"
    
    # Se a safra começa em Outubro e termina em Março, o ano final é o seguinte.
    if mes_fim < mes_inicio:
        ano_fim_real = ano_fim + 1
    else:
        ano_fim_real = ano_fim
        
    data_fim_str = f"{ano_fim_real}-{mes_fim:02d}-28"
    
    # --- DEFINIÇÃO DAS COLEÇÕES DE IMAGENS ---
    # Sentinel-2: Imagens ópticas (visão humana + infravermelho)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometria).filterDate(data_inicio_str, data_fim_str)
    # MODIS: Satélite diário de baixa resolução, ótimo para temperatura (LST)
    modis = ee.ImageCollection("MODIS/061/MOD11A2").filterBounds(geometria).filterDate(data_inicio_str, data_fim_str)
    # CHIRPS: Dados de chuva interpolados via satélite + estações terrestres
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(geometria).filterDate(data_inicio_str, data_fim_str)
    
    # Sentinel-1 (Radar): Funciona dia e noite e atravessa nuvens. Crítico para safras chuvosas.
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(geometria).filterDate(data_inicio_str, data_fim_str) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
           .filter(ee.Filter.eq('instrumentMode', 'IW')) \
           .map(processar_radar_s1)
           
    # SMAP: Satélite da NASA específico para umidade do solo.
    smap = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005').filterBounds(geometria).filterDate(data_inicio_str, data_fim_str) \
             .map(processar_smap)

    # Função interna processada NO SERVIDOR DO GOOGLE para cada intervalo de tempo (ex: quinzenal)
    def processar_intervalo(data_milis):
        inicio = ee.Date(data_milis)
        fim = inicio.advance(passo_dias, 'day')
        
        # 1. Processamento Sentinel-2 (Óptico)
        def mascara_nuvens(img):
            # Banda QA60 contém bits de qualidade. Bit 10 = nuvem opaca, Bit 11 = cirrus.
            qa = img.select('QA60')
            mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
            return img.updateMask(mask).divide(10000) # Divide por 10k para ter refletância 0-1
            
        s2_filtrado = s2.filterDate(inicio, fim).map(mascara_nuvens)
        
        # Métrica de confiança: Se não houver imagens limpas no período, confiança é 0.
        confianca = ee.Algorithms.If(s2_filtrado.size().gt(0), 1.0, 0.0)

        def indices_s2():
            # Cria um mosaico (mediana dos pixels) para remover ruídos residuais
            mosaico = s2_filtrado.median()
            ndvi = mosaico.normalizedDifference(['B8', 'B4']).rename('NDVI') # Infravermelho Próximo - Vermelho
            ndmi = mosaico.normalizedDifference(['B8', 'B11']).rename('NDMI') # Infravermelho Próximo - SWIR
            return ndvi.addBands(ndmi)

        def indices_vazios():
            # Cria imagem "dummy" vazia para não quebrar o código se o período for 100% nublado
            dummy = ee.Image.constant(0).selfMask()
            return dummy.rename('NDVI').addBands(dummy.rename('NDMI'))

        img_indices = ee.Image(ee.Algorithms.If(
            s2_filtrado.size().gt(0), indices_s2(), indices_vazios()
        ))
        
        # 2. Processamento MODIS (Temperatura)
        modis_fil = modis.filterDate(inicio, fim)
        lst = ee.Image(ee.Algorithms.If(
            modis_fil.size().gt(0),
            # Conversão Kelvin para Celsius: (Valor * 0.02) - 273.15
            modis_fil.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).rename('LST'),
            ee.Image.constant(0).selfMask().rename('LST')
        ))
        
        # 3. Processamento Chuva (CHIRPS)
        chirps_fil = chirps.filterDate(inicio, fim)
        precip = ee.Image(ee.Algorithms.If(
            chirps_fil.size().gt(0),
            # Usamos MEAN (média mm/dia) em vez de SUM para ser comparável entre períodos de tamanhos diferentes
            chirps_fil.mean().rename('precip_mm_day'), 
            ee.Image.constant(0).rename('precip_mm_day')
        ))

        # 4. Processamento Radar (Sentinel-1)
        s1_fil = s1.filterDate(inicio, fim)
        s1_comp = ee.Image(ee.Algorithms.If(
            s1_fil.size().gt(0),
            s1_fil.select(['VV', 'VH']).mean(),
            ee.Image.constant([0, 0]).rename(['VV', 'VH']).selfMask() 
        ))

        # 5. Processamento Umidade Solo (SMAP)
        smap_fil = smap.filterDate(inicio, fim)
        smap_comp = ee.Image(ee.Algorithms.If(
            smap_fil.size().gt(0),
            smap_fil.select('ssm').mean(),
            ee.Image.constant(0).rename('ssm').selfMask()
        ))
        
        # Empilha todas as bandas em uma única imagem virtual
        img_final = ee.Image.cat([img_indices, lst, precip, s1_comp, smap_comp]) \
                .set('system:time_start', inicio.millis()) \
                .set('confianca_s2', confianca)
        
        # Extrai a média estatística de todos os pixels dentro do polígono desenhado
        stats = img_final.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometria,
            scale=escala,
            bestEffort=True,
            maxPixels=1e9
        )
        return ee.Feature(None, stats).set('date', inicio.format('YYYY-MM-dd')).set('confianca_s2', confianca)

    # Gera a sequência de datas e mapeia a função de processamento
    lista_dias = ee.List.sequence(ee.Date(data_inicio_str).millis(), ee.Date(data_fim_str).millis(), passo_dias * 24 * 3600 * 1000)
    colecao_resultados = ee.FeatureCollection(lista_dias.map(processar_intervalo))
    
    try:
        # Transfere os dados da nuvem (Google) para o cliente (Python)
        dados_lista = colecao_resultados.getInfo()['features']
        df = pd.DataFrame([feat['properties'] for feat in dados_lista])
    except Exception as e:
        st.error(f"Erro na comunicação com GEE: {e}")
        return pd.DataFrame()
    
    return df

@st.cache_data(ttl=3600)
def buscar_datas_imagens(coords_poligono, ano_inicio, ano_fim):
    """
    Busca metadados das imagens disponíveis para alimentar o seletor de visualização.
    Filtra imagens com mais de 60% de nuvens para não poluir a lista.
    """
    geo = ee.Geometry.Polygon(coords_poligono)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
           .filterBounds(geo)\
           .filterDate(f"{ano_inicio}-01-01", f"{ano_fim}-12-31")\
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))\
           .select(['QA60']) # Seleciona leve para performance
           
    info = s2.map(lambda img: ee.Feature(None, {
        'id': img.get('system:id'), 
        'date': img.date().format('YYYY-MM-dd'), 
        'cloud': img.get('CLOUDY_PIXEL_PERCENTAGE')
    })).getInfo()
    
    lista = []
    if 'features' in info:
        for f in info['features']:
            p = f['properties']
            lista.append({
                'display': f"{p['date']} (☁️ {p['cloud']:.1f}%)", 
                'id': p['id'], 
                'date': p['date']
            })
            
    lista.sort(key=lambda x: x['date'], reverse=True)
    return lista

def pre_processar_dados(df):
    """Limpeza e interpolação de dados faltantes."""
    if df.empty: return df
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Identifica a safra agrícola (Ano Agrícola). Se passar de Julho, considera safra seguinte.
    df['safra'] = df['date'].apply(lambda d: f"{d.year}/{d.year+1}" if d.month >= 7 else f"{d.year-1}/{d.year}")
    
    # Interpolação: Se houver buracos nos dados (ex: nuvens por 1 mês), preenche com média linear.
    cols_numericas = df.select_dtypes(include=np.number).columns
    df[cols_numericas] = df[cols_numericas].interpolate(method='linear').ffill().bfill()
    
    # Garante estrutura consistente do DataFrame
    for col in VARIAVEIS:
        if col not in df.columns: df[col] = 0.0
    if 'confianca_s2' not in df.columns: df['confianca_s2'] = 1.0
            
    return df

def selecionar_safras_normais(df_prod, percentil_corte=0.15):
    """
    Identifica anos 'normais' para criar o Baseline (Régua de Comparação).
    Remove anos de produtividade muito alta ou muito baixa (outliers), 
    para que o índice de estresse não seja distorcido por eventos extremos.
    """
    mediana = df_prod['produtividade_kg_ha'].median()
    lim_inf = mediana * (1 - percentil_corte)
    lim_sup = mediana * (1 + percentil_corte)
    return df_prod[(df_prod['produtividade_kg_ha'] > lim_inf) & (df_prod['produtividade_kg_ha'] < lim_sup)]['safra'].tolist()

def calcular_estatisticas_base(df, safras_normais, features):
    """Calcula o 'Comportamento Esperado' (Média/Std) para cada quinzena do ano."""
    df_base = df[df['safra'].isin(safras_normais)].copy()
    if df_base.empty: return None
    
    df_base['quinzena'] = df_base['date'].dt.dayofyear // 15
    # Agrupa por quinzena para ter o perfil esperado de cada época do ano
    stats = df_base.groupby('quinzena')[features].agg(['mean', 'std'])
    return stats

def calcular_ieha_robusto(df, stats_base, pesos_dinamicos, features):
    """
    Cálculo do Índice de Estresse Hídrico Agronômico (IEHA).
    1. Calcula Z-Score (o quanto o dado atual desvia do normal).
    2. Verifica tendência (Momentum) de piora/melhora.
    3. Aplica pesos definidos pela IA para cada variável.
    4. Realiza fallback: Se tiver nuvem, ignora NDVI e usa Radar.
    """
    if df.empty or stats_base is None: return df
    
    df_final = df.copy()
    df_final['quinzena'] = df_final['date'].dt.dayofyear // 15
    
    valores_ieha = []
    lista_status = []
    vars_opticas = ['NDVI', 'NDMI']

    # Pré-cálculo de Z-Scores e Tendências (Momentum)
    for f in features:
        mapa_media = stats_base[(f, 'mean')].to_dict()
        mapa_std = stats_base[(f, 'std')].to_dict()
        
        # Compara valor atual com a média histórica daquela quinzena
        media_ref = df_final['quinzena'].map(mapa_media).fillna(df_final[f].mean())
        std_ref = df_final['quinzena'].map(mapa_std).fillna(1.0)
        df_final[f'{f}_z'] = (df_final[f] - media_ref) / std_ref
        
        # Calcula a derivada (velocidade de mudança) dos últimos 3 períodos
        df_final[f'{f}_trend'] = df_final[f'{f}_z'].rolling(window=3, min_periods=1).mean().diff().fillna(0)

    for idx, row in df_final.iterrows():
        conf_optica = row.get('confianca_s2', 1.0)
        
        pesos_atuais = pesos_dinamicos.copy()
        msg_extra = ""
        
        # Lógica de Fallback: Se a confiança óptica for baixa (< 30% por causa de nuvens), zera o peso do NDVI óptico e usa Radar
        if conf_optica < 0.3:
            for opt in vars_opticas: pesos_atuais[opt] = 0.0
            msg_extra = " (☁️ Radar/SMAP)"

        soma_pontos = 0
        soma_pesos = 0
        contribuicoes = {}
        
        for f, peso in pesos_atuais.items():
            z_val = row.get(f'{f}_z', 0.0)
            if pd.isna(z_val): z_val = 0.0
            
            contrib = z_val * peso
            soma_pontos += contrib
            soma_pesos += abs(peso)
            contribuicoes[f] = abs(contrib)
        
        # Cálculo Final do Índice (0 a 10)
        z_medio = soma_pontos / (soma_pesos if soma_pesos > 0 else 1.0)
        ieha = 5 + (z_medio * 2.5)
        ieha = np.clip(ieha, 0, 10)
        valores_ieha.append(ieha)
        
        # Diagnóstico de Causa e Prognóstico
        status = "NORMAL"
        if ieha > risco_limite*ajuste_atencao:
            nivel = "ALERTA" if ieha > risco_limite else "ATENÇÃO"
            causa = max(contribuicoes, key=contribuicoes.get) if contribuicoes else "N/A"
            tendencia = row.get(f'{causa}_trend', 0.0)
            peso_causa = pesos_atuais.get(causa, 0.0)
            
            # Se peso é negativo (chuva) e tendência cai -> Piora
            # Se peso é positivo (temperatura) e tendência sobe -> Piora
            agravando = (peso_causa < 0 and tendencia < -0.05) or (peso_causa > 0 and tendencia > 0.05)
            melhorando = (peso_causa < 0 and tendencia > 0.05) or (peso_causa > 0 and tendencia < -0.05)
            
            prog = ""
            if agravando: prog = " (Piora Recente 📈)"
            elif melhorando: prog = " (Melhora Recente 📉)"
            else: prog = " (Estável)"
            
            status = f"{nivel} (Causa: {causa}){prog}{msg_extra}"
        else:
            status = f"NORMAL{msg_extra}"
            
        lista_status.append(status)

    df_final['IEHA'] = valores_ieha
    df_final['Status'] = lista_status
    return df_final

def treinar_modelo_obter_pesos(df, df_prod_hist):
    """
    Usa Machine Learning (Random Forest) para descobrir quais variáveis
    impactam mais a produtividade NESTA região específica.
    Retorna os 'Pesos' personalizados para o cálculo do IEHA.
    
    Treina Random Forest para:
    1. Prever produtividade.
    2. Calcular importância das variáveis (Pesos).
    3. Retornar score de confiança (R2).
    """
    if df.empty: return None, {}, 0, 0.0
    
    colunas_validas = [c for c in VARIAVEIS if c in df.columns]
    
    # Agrega dados por safra para cruzar com o histórico anual do IBGE
    stats_safra = df.groupby('safra')[colunas_validas].mean().reset_index()
    dados_treino = pd.merge(stats_safra, df_prod_hist, on='safra', how='inner')
    
    media_hist = df_prod_hist['produtividade_kg_ha'].mean()

    if len(dados_treino) < 2: 
        return None, {f: 1.0/len(colunas_validas) for f in colunas_validas}, media_hist, 0.0
        
    X = dados_treino[colunas_validas]
    y = dados_treino['produtividade_kg_ha']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X.values, y)
    
    # Score R2 (Coeficiente de Determinação): Quão bem o modelo explica os dados (0 a 1)
    score_r2 = modelo.score(X.values, y)
    
    sinais = {'NDVI': -1, 'NDMI': -1, 'LST': 1, 'precip_mm_day': -1, 'ssm': -1, 'VV': -1, 'VH': -1}
    importancias = modelo.feature_importances_
    pesos = {}
    for i, col in enumerate(colunas_validas):
        s = sinais.get(col, -1)
        pesos[col] = importancias[i] * s 
        
    return modelo, pesos, media_hist, score_r2

def prever_tendencia_futura(df_completo, mes_colheita, passo_dias=15):
    """
    Projeta o futuro baseando-se na sazonalidade aprendida.
    Retorna também o Score R2 da aderência sazonal.
    """
    if df_completo.empty: return pd.DataFrame(), 0.0
    
    df_treino = df_completo[['date', 'IEHA']].copy().dropna()
    if df_treino.empty: return pd.DataFrame(), 0.0

    df_treino['dia_do_ano'] = df_treino['date'].dt.dayofyear
    X, y = df_treino[['dia_do_ano']], df_treino['IEHA']
    
    # Modelo aprende o 'formato' do ano (ex: estresse sobe em agosto, cai em novembro)
    modelo_sazonal = RandomForestRegressor(n_estimators=100, random_state=42).fit(X.values, y)
    score_sazonal = modelo_sazonal.score(X.values, y) # Confiança na sazonalidade
    
    ultima_data = df_completo['date'].max()
    ano_alvo = ultima_data.year + 1 if mes_colheita < ultima_data.month else ultima_data.year
    data_limite = datetime(ano_alvo, mes_colheita, 28)
    
    if data_limite <= ultima_data: return pd.DataFrame(), score_sazonal
    
    datas_futuras = [ultima_data + timedelta(days=x) for x in range(passo_dias, (data_limite - ultima_data).days + passo_dias, passo_dias)]
    df_futuro = pd.DataFrame({'date': datas_futuras})
    df_futuro['dia_do_ano'] = df_futuro['date'].dt.dayofyear
    
    pred_base = modelo_sazonal.predict(df_futuro[['dia_do_ano']])
    
    # Bias Correction: Se hoje está pior que a média, o futuro começa pior também
    valor_atual = df_completo['IEHA'].iloc[-1]
    pred_para_hoje = modelo_sazonal.predict([[ultima_data.dayofyear]])[0]
    vies_atual = (valor_atual - pred_para_hoje) if not pd.isna(valor_atual) else 0.0
    
    df_futuro['IEHA_Pred'] = np.clip(pred_base + vies_atual, 0, 10)
    df_futuro['Tipo'] = 'Previsão IA (Sazonal)'
    
    # Calcula inclinação da reta futura para resumo textual
    inclinacao = np.polyfit(range(len(df_futuro)), df_futuro['IEHA_Pred'], 1)[0]
    if inclinacao < -0.05: msg = " (Tendência de Melhora)"
    elif inclinacao > 0.05: msg = " (Risco Agravado)"
    else: msg = " (Estável)"
    
    df_futuro['Msg_Tendencia'] = msg
    
    return df_futuro, score_sazonal

# --- INTERFACE STREAMLIT (SIDEBAR) ---
st.sidebar.title("Configurações")
st.sidebar.header("🗓️ Período de Análise")
ano_ini = st.sidebar.number_input("Ano Início", 2018, 2024, 2018)
ano_fin = st.sidebar.number_input("Ano Fim", 2018, 2026, 2025)
c_m1, c_m2 = st.sidebar.columns(2)
mes_ini = c_m1.selectbox("Mês Plantio", range(1, 13), index=9)
mes_fim = c_m2.selectbox("Mês Colheita", range(1, 13), index=3)

st.sidebar.header("⚙️ Parâmetros")
risco_limite = st.sidebar.slider("Risco Aceitável (IEHA)", 0.0, 10.0, 7.5)
passo_dias = st.sidebar.number_input("Passo (Dias)", 5, 30, 15)
escala_m = st.sidebar.number_input("Escala (m)", 10, 500, 100)

st.sidebar.subheader("📋 Dados de Treino")
df_prod_edit = st.sidebar.data_editor(DADOS_HISTORICOS_PADRAO, num_rows="dynamic")

# --- UI PRINCIPAL ---
st.title("Sentinela Agroclimática")

col_mapa, col_acao = st.columns([2, 1])

with col_mapa:
    st.subheader("1. Definição do Talhão")
    m = geemap.Map(center=[LAT_INICIAL, LON_INICIAL], zoom=13, basemap="HYBRID")    
    mapa_out = m.to_streamlit(height=450, bidirectional=True)
    
    coords_desenho = None
    if mapa_out and 'last_active_drawing' in mapa_out and mapa_out['last_active_drawing']:
        coords_desenho = mapa_out['last_active_drawing']['geometry']['coordinates'][0]
    
    if coords_desenho:
        with st.expander("📍 Coordenadas GeoJSON"): 
            st.code(coords_desenho, language="json")

with col_acao:
    st.markdown("### Execução")
    btn_rodar = st.button("🚀 Rodar Análise", type="primary")
    
    if 'metricas_result' in st.session_state:
        m = st.session_state['metricas_result']
        st.divider()
        st.metric("Safra Analisada", m['safra'])
        st.metric("IEHA Atual", m['ieha_fmt'], delta=m.get('status_curto', ''), delta_color="inverse")
        
        if m['ieha_val'] > risco_limite:
            st.error(f"⚠️ **RISCO ELEVADO**\n\n{m.get('status_completo','')}")
        else:
            st.success(f"✅ **Monitoramento Normal**")

# --- ORQUESTRAÇÃO DO PROCESSAMENTO ---
if btn_rodar and coords_desenho:
    with st.spinner('1/5 Coletando dados Multissensor (S2, S1, SMAP, MODIS, CHIRPS)...'):
        df_bruto = coletar_dados_gee(coords_desenho, ano_ini, ano_fin, mes_ini, mes_fim, passo_dias, escala_m)
        lista_imgs = buscar_datas_imagens(coords_desenho, ano_ini, ano_fin)

    if not df_bruto.empty:
        with st.spinner('2/5 Pré-processamento e Baseline Sazonal...'):
            df_limpo = pre_processar_dados(df_bruto)
            anos_normais = selecionar_safras_normais(df_prod_edit)
            stats_baseline = calcular_estatisticas_base(df_limpo, anos_normais, VARIAVEIS)
            
        with st.spinner('3/5 Treinando IA para Pesos Dinâmicos...'):
            # Captura também o score R2 do modelo
            modelo_prod, pesos, media_hist, score_r2_prod = treinar_modelo_obter_pesos(df_limpo, df_prod_edit)
            
        with st.spinner('4/5 Calculando IEHA Robusto (Diagnóstico & Prognóstico)...'):
            # Usa apenas dados do passado para calcular o histórico
            df_passado = df_limpo[df_limpo['date'] <= pd.Timestamp.now()]
            df_final = calcular_ieha_robusto(df_passado, stats_baseline, pesos, VARIAVEIS)
            
            # Estimativa de Produtividade (Safra Atual)
            safra_atual = df_final['safra'].iloc[-1]
            df_safra_atual = df_final[df_final['safra'] == safra_atual]
            predicao_kg = 0
            if not df_safra_atual.empty:
                cols_modelo = [c for c in VARIAVEIS if c in df_safra_atual.columns]
                x_pred = pd.DataFrame([df_safra_atual[cols_modelo].mean()], columns=cols_modelo)
                try: predicao_kg = modelo_prod.predict(x_pred)[0]
                except: pass
            
            # Captura também o score R2 da tendência sazonal
            df_futuro, score_r2_sazonal = prever_tendencia_futura(df_final, mes_fim, passo_dias)
            msg_tendencia = df_futuro['Msg_Tendencia'].iloc[0] if not df_futuro.empty else ""
            
            status_full = df_final['Status'].iloc[-1]
            status_short = status_full.split('(')[0] if '(' in status_full else status_full

            # Salva no estado da sessão (Persistência)
            st.session_state.update({
                'dados_final': df_final, 
                'dados_futuro': df_futuro, 
                'imagens_disp': lista_imgs, 
                'coords': coords_desenho, 
                'stats_baseline': stats_baseline, # Salva para exibir no expander
                'pesos_ia': pesos, # Salva para exibir no gráfico
                'idx_img': 0,
                'metricas_result': {
                    'safra': safra_atual, 
                    'ieha_val': df_final['IEHA'].iloc[-1],
                    'ieha_fmt': f"{df_final['IEHA'].iloc[-1]:.2f}",
                    'pred': predicao_kg, 
                    'media_hist': media_hist, 
                    'msg_tendencia': msg_tendencia,
                    'status_completo': status_full, 
                    'status_curto': status_short,
                    'score_prod': score_r2_prod,   # Novo: Confiança Produtividade
                    'score_saz': score_r2_sazonal  # Novo: Confiança Tendência
                }
            })
            st.rerun()
    else:
        st.error("Não foi possível coletar dados para essa região/período.")

# --- EXIBIÇÃO DOS RESULTADOS ---
if 'dados_final' in st.session_state:
    df, df_fut = st.session_state['dados_final'], st.session_state['dados_futuro']
    res = st.session_state['metricas_result']
    
    st.divider()
    st.subheader("2. Monitoramento Temporal (IEHA - Histórico)")
    
    fig = px.line(df, x='date', y='IEHA', color='safra', hover_data=['Status'], markers=True)
    fig.add_hline(y=risco_limite*0.8, line_dash="dash", line_color="orange", annotation_text="Atenção")
    fig.add_hline(y=risco_limite, line_dash="dash", line_color="red", annotation_text="Alerta")
    st.plotly_chart(fig, width='stretch')
    
    st.divider()
    st.subheader("3. IA & Futuro: Produtividade e Tendência IEHA")
    c_ia1, c_ia2 = st.columns(2)    
    
    with c_ia1:
        st.markdown("#### Estimativa de Colheita (Random Forest)")
        st.caption("Usa: NDVI, LST, Chuva, Radar (S1) e Umidade Solo (SMAP)")
        delta_pct = (res['pred'] - res['media_hist']) / res['media_hist'] * 100
        st.metric("Produtividade Prevista", f"{res['pred']:.0f} kg/ha", f"{delta_pct:+.1f}% vs Média")
        st.metric("Média Histórica (Total)", f"{res['media_hist']:.0f} kg/ha")
        # Exibe a confiança do modelo
        st.caption(f"🎯 Aderência do Modelo (R²): **{res['score_prod']*100:.1f}%**")
        
    with c_ia2:
        st.markdown(f"#### Tendência Futura {res['msg_tendencia']}")
        st.caption("Curva de tendência aprendida com ajuste de viés atual")
        if not df_fut.empty:
            df_plot = pd.concat([
                df[df['safra']==res['safra']][['date','IEHA','Status']].assign(Tipo='Observado'),
                df_fut[['date','IEHA_Pred','Msg_Tendencia']].rename(columns={'IEHA_Pred':'IEHA'}).assign(Tipo='Projeção IA')
            ])
            fig2 = px.line(df_plot, x='date', y='IEHA', color='Tipo', markers=True, 
                           color_discrete_map={'Observado':'blue', 'Projeção IA':'orange'})
            fig2.update_traces(selector={'legendgroup':'Projeção IA'}, line=dict(dash='dot'))
            fig2.add_hline(y=risco_limite, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, width='stretch')
            # Exibe a confiança da tendência
            st.caption(f"🎯 Confiabilidade da Sazonalidade (R²): **{res['score_saz']*100:.1f}%**")
        else: 
            st.info("Sem projeção disponível (fim de safra ou dados insuficientes).")

    st.divider()
    st.subheader("4. Inspeção Visual (Drill-Down)")
    imgs = st.session_state['imagens_disp']
    
    if imgs:
        opcoes_txt = [x['display'] for x in imgs]
        
        if 'sb_sel' not in st.session_state:
            st.session_state.sb_sel = opcoes_txt[0]

        def atualizar_idx():
            try: st.session_state.idx_img = opcoes_txt.index(st.session_state.sb_sel)
            except: st.session_state.idx_img = 0
            
        def voltar(): 
            st.session_state.idx_img = min(len(imgs)-1, st.session_state.idx_img + 1)
            st.session_state.sb_sel = opcoes_txt[st.session_state.idx_img]
            
        def avancar(): 
            st.session_state.idx_img = max(0, st.session_state.idx_img - 1)
            st.session_state.sb_sel = opcoes_txt[st.session_state.idx_img]

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Data:", opcoes_txt, key="sb_sel", on_change=atualizar_idx)
            b1, b2 = st.columns(2)
            b1.button("◀ Anterior", on_click=voltar, width='stretch')
            b2.button("Próximo ▶", on_click=avancar, width='stretch')
            st.markdown("---")
            camada = st.radio("Camada:", ["RGB Real", "NDVI (Vegetação)", "Estresse (NDWI)"], horizontal=True)
            
        with c2:
            if st.session_state.coords:
                selecao = imgs[st.session_state.idx_img]
                geo_poly = ee.Geometry.Polygon(st.session_state.coords)
                img_ee = ee.Image(selecao['id'])
                
                # Pinta o contorno do talhão de vermelho
                outline = ee.Image().byte().paint(ee.FeatureCollection([geo_poly]), 1, 3).visualize(palette=['red'])
                
                # Máscara de nuvens para não atrapalhar a visualização
                qa = img_ee.select('QA60')
                mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
                
                if "RGB" in camada:
                    viz = img_ee.select(['B4','B3','B2']).divide(10000).visualize(min=0, max=0.3, gamma=1.4)
                elif "NDVI" in camada:
                    ndvi = img_ee.divide(10000).normalizedDifference(['B8','B4'])
                    viz = ndvi.visualize(min=0, max=0.8, palette=['red', 'yellow', 'green'])
                else:
                    viz = img_ee.divide(10000).normalizedDifference(['B8','B11']).visualize(min=-0.5, max=0.5, palette=['red','yellow','blue'])
                
                final = viz.blend(outline).clip(geo_poly.buffer(500))
                
                try:
                    url = final.getThumbURL({'dimensions': 800, 'region': geo_poly.buffer(100).bounds(), 'format': 'png'})
                    st.image(url, caption=f"Talhão em {selecao['date']}", width='stretch')
                except: 
                    st.error("Erro ao gerar imagem desta data.")
                    
    st.divider()
    st.subheader("5. Transparência de Dados")
    
    # --- EXPANDERS DE TRANSPARÊNCIA DE DADOS ---
    with st.expander("Detalhes do Processamento e Inteligência"):
        st.markdown("""
        **Como o Sentinela funciona:**
        1.  **Coleta Multissensor:** Buscamos dados de 5 satélites diferentes.
        2.  **IA de Pesos:** O algoritmo aprende quais variáveis impactam mais a produtividade *nesta região* (veja o gráfico abaixo).
        3.  **IEHA:** O Índice de Estresse Hídrico Agronômico combina esses dados. Se houver nuvens, o sistema foca no Radar.
        """)
        
        # Gráfico de Importância das Variáveis (Pesos)
        if 'pesos_ia' in st.session_state:
            pesos = st.session_state['pesos_ia']
            # Converte para DataFrame para o Plotly, usando valores absolutos para mostrar magnitude da importância
            df_pesos = pd.DataFrame({
                'Sensor': list(pesos.keys()),
                'Importância (Peso)': [abs(v) for v in pesos.values()],
                'Impacto': ['Negativo' if v < 0 else 'Positivo' for v in pesos.values()]
            }).sort_values('Importância (Peso)', ascending=True)
            
            fig_pesos = px.bar(
                df_pesos, x='Importância (Peso)', y='Sensor', color='Impacto', 
                orientation='h', title="O que mais impactou a análise da IA?",
                color_discrete_map={'Negativo': 'red', 'Positivo': 'blue'}
            )
            st.plotly_chart(fig_pesos, width='stretch')
        
        st.markdown("""
        **Glossário:**
        1. LST (Land Surface Temperature): Temperatura da superfície (estresse térmico).
        2. VV, VH: Polarizações do Radar (Sentinel-1). Medem a estrutura física da planta e rugosidade, sendo essenciais quando há nuvens bloqueando os sensores ópticos.
        3. NDVI (Normalized Difference Vegetation Index): Mede o vigor da planta (cor verde).
        4. NDMI (Normalized Difference Moisture Index): Mede o conteúdo de água na folha.
        5. precip_mm_day: Precipitação média diária (chuva).
        6. ssm (Surface Soil Moisture): Umidade do solo superficial (satélite SMAP).
        """)

    with st.expander("Dados Brutos do Baseline Sazonal (O 'Normal' de cada Quinzena)"):
        st.write("Estas são as médias e desvios padrão históricos usados para calcular o Z-Score.")
        if 'stats_baseline' in st.session_state and st.session_state['stats_baseline'] is not None:
            st.dataframe(st.session_state['stats_baseline'])
        else:
            st.info("Baseline não calculado.")

    with st.expander("Tabela Final Processada"):
        st.dataframe(df)
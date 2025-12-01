# 🛰️ Sentinela Agroclimática

O **Sentinela Agroclimática** é uma ferramenta de monitoramento agrícola baseada em Python, Streamlit e Google Earth Engine (GEE). O sistema utiliza **Inteligência Artificial (Random Forest)** para fundir dados de múltiplos satélites e estimar a produtividade, detectar estresse hídrico e prever tendências sazonais, mesmo em condições de alta cobertura de nuvens.

-----

## 🎯 Contexto e Objetivo

A agricultura de precisão enfrenta desafios constantes com a variabilidade climática e a falta de dados visuais em períodos chuvosos. Este projeto resolve esses problemas integrando:

1.  **Fusão de Sensores:** Combina dados ópticos (Sentinel-2), térmicos (MODIS), radar (Sentinel-1), precipitação (CHIRPS) e umidade do solo (SMAP).
2.  **Resiliência a Nuvens:** Quando os satélites ópticos estão bloqueados por nuvens, o sistema automaticamente "rebaixa" o peso dessas variáveis e prioriza dados de radar (que atravessam nuvens), garantindo monitoramento contínuo.
3.  **Machine Learning Personalizado:** O sistema treina um modelo exclusivo para a região desenhada pelo usuário, aprendendo quais variáveis ambientais mais impactaram a produtividade histórica daquela fazenda específica.

-----

## 🚀 Funcionalidades Principais

### 1\. Definição Geoespacial

  * Interface interativa para desenhar talhões/polígonos sobre o mapa.
  * Extração automática de coordenadas para consulta na nuvem do Google.

### 2\. Análise Multissensor (Big Data)

O sistema coleta e processa automaticamente séries temporais de:

  * **Sentinel-2:** Índices de vegetação (NDVI) e umidade foliar (NDMI).
  * **Sentinel-1 (Radar):** Estrutura da planta e rugosidade (Bandas VV e VH) — funciona dia e noite, com ou sem chuva.
  * **MODIS:** Temperatura da Superfície Terrestre (LST) para detectar estresse térmico.
  * **CHIRPS:** Dados de precipitação diária.
  * **SMAP:** Umidade superficial do solo (NASA).

### 3\. Inteligência Artificial (IA)

  * **Pesos Dinâmicos:** Um algoritmo *Random Forest Regressor* analisa o histórico de produtividade inserido pelo usuário e define, matematicamente, qual fator (chuva, temperatura, vigor) é mais crítico para aquela cultura.
  * **Estimativa de Produtividade:** Previsão de colheita em kg/ha baseada nas condições atuais comparadas ao modelo treinado.
  * **Projeção Sazonal:** Previsão de tendência futura do estresse (melhora ou piora) baseada em padrões sazonais aprendidos.

### 4\. Índice IEHA

Cálculo do **Índice de Estresse Hídrico Agronômico (IEHA)**, uma métrica composta que varia de 0 a 10, normalizada pelo Z-Score histórico da região (compara o atual com o "normal" esperado para a quinzena).

### 5\. Visualização

  * Gráficos interativos (Plotly) de evolução temporal.
  * Visualizador de imagens de satélite (RGB, NDVI, NDWI) com seletor de datas.
  * Gráfico de importância de variáveis (Explainable AI).

-----

## 🛠️ Pré-requisitos e Instalação

Para executar este projeto localmente, você precisará dos seguintes softwares e bibliotecas:

### 1\. Linguagem e Ambiente

  * **Python 3.9 ou superior.**
  * Recomendado o uso de ambiente virtual (`venv` ou `conda`).

### 2\. Conta no Google Earth Engine

O sistema roda processamento pesado nos servidores do Google.

  * Cadastre-se em: [code.earthengine.google.com](https://code.earthengine.google.com/).
  * Crie um "Cloud Project" no Google Cloud Platform associado à sua conta.

### 3\. Instalação das Dependências

Instale as dependências via terminal:

```bash
pip install -r requirements.txt
```

### 4\. Configuração de Autenticação

Antes de rodar o app, autentique seu ambiente Python com o Google Earth Engine:

```bash
earthengine authenticate
```

*Siga as instruções no navegador para autorizar o acesso.*

> **Nota:** No código (`app.py`), localize a variável `MEU_PROJETO_ID = "projetoaero"` e altere `"projetoaero"` para o ID do seu projeto no Google Cloud.

-----

## 💻 Como Executar

1.  Abra o terminal na pasta do projeto.
2.  Execute o comando do Streamlit:
    ```bash
    streamlit run app.py
    ```
3.  O navegador abrirá automaticamente (geralmente em `http://localhost:8501`).

-----

## 📖 Guia de Uso Rápido

1.  **Barra Lateral:** Defina o intervalo de datas (Safra) e ajuste os dados históricos de produtividade (gabarito para a IA treinar), de preferência usando fontes confiáveis como a [tabela 5457 do IBGE](https://sidra.ibge.gov.br/tabela/5457).
2.  **Mapa:** Use as ferramentas de desenho (retângulo ou polígono) para marcar a área da lavoura que deseja analisar.
3.  **Processamento:** Clique no botão **"🚀 Rodar Análise"**.
4.  **Resultados:**
      * Verifique o "IEHA Atual" e o status de Alerta/Atenção.
      * Analise o gráfico de tendência temporal.
      * Consulte a previsão de produtividade da IA.
      * No final da página, explore as imagens de satélite visuais para validar os dados.

-----

## 📚 Glossário Técnico

| Variável | Descrição | Satélite |
| :--- | :--- | :--- |
| **NDVI** | Índice de Vegetação (Vigor/Verdor da planta). | Sentinel-2 |
| **NDMI** | Índice de Umidade (Conteúdo de água na folha). | Sentinel-2 |
| **LST** | Temperatura da Superfície (Land Surface Temperature). | MODIS |
| **VV / VH** | Retrospectiva do Radar (Estrutura física e biomassa). | Sentinel-1 |
| **SSM** | Umidade do Solo Superficial. | SMAP |
| **Z-Score** | Medida estatística que indica quantos desvios-padrão o valor atual está longe da média histórica. | Calculado |

-----

## ⚠️ Notas Importantes

  * **Cache:** O sistema utiliza cache (`ttl=3600`) para economizar requisições ao Google. Se mudar o polígono, o cache recarrega automaticamente.
  * **Limites do GEE:** Polígonos muito grandes ou períodos muito longos (ex: 10 anos) podem exceder o tempo limite de processamento da API do Google (Computation Time Out).
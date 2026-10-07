# 📈 iwannaberich: Quantitative Market Intelligence & Screening Engine

[![CI](https://github.com/fmpalmab/iwannaberich/actions/workflows/ci.yml/badge.svg)](https://github.com/fmpalmab/iwannaberich/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: standard](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://pep8.org/)

Suite moderna de ingeniería cuantitativa y procesamiento de señales financieras orientada al mercado bursátil chileno (**Bolsa de Santiago - Tickers `.SN`**). 

Procesa cientos de instrumentos de renta variable y fondos, filtra por liquidez real transada en CLP, estima métricas de riesgo estocástico y de cola (VaR, CVaR, Curtosis, Drawdown), detecta señales técnicas sistemáticas y exporta reportes ejecutivos automatizados.

---

## 🏗️ Arquitectura del Sistema

El proyecto ha sido refactorizado como un paquete modular `iwannaberich` bajo especificaciones modernas (PEP 517/621):

```
iwannaberich/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline de CI (Pytest en 3.10, 3.11, 3.12)
├── iwannaberich/                 # Paquete principal
│   ├── __init__.py              # Exportaciones públicas del paquete
│   ├── __main__.py              # Punto de entrada ejecutable (python -m iwannaberich)
│   ├── cli.py                   # Interfaz de línea de comandos unificada
│   ├── config.py                # Rutas y constantes financieras (Risk Free Rate, liquidez)
│   ├── indicators.py            # Indicadores técnicos (RSI Wilder, SMA, Bandas Bollinger, MACD)
│   ├── loader.py                # Ingesta vía yfinance y generador sintético Jump-Diffusion
│   ├── metrics.py               # Ratios de eficiencia (Sharpe, Sortino, Calmar) y riesgo de cola (VaR, CVaR)
│   ├── parser.py                # Parser regex y clasificador de activos (.SN)
│   ├── pipeline.py              # Orquestador end-to-end del flujo cuantitativo
│   ├── processor.py             # Consolidación Parquet y filtro de liquidez bursátil
│   ├── report.py                # Generador de reportes ejecutivos de inteligencia
│   ├── scanner.py               # Motor de reglas de trading (Dip, Oversold, Momentum, Breakout)
│   └── visualizer.py            # Renderizado gráfico de análisis técnico (Price + Bollinger + RSI)
├── data/
│   ├── raw/precios_historicos/  # Parquets históricos individuales por ticker
│   ├── processed/               # Precios consolidados limpios y matrices estadísticas
│   └── reporte_destacados.txt   # Reporte ejecutivo generado
├── src/                         # Shims de compatibilidad hacia atrás (scripts 01-06)
├── tests/                       # Suite de pruebas unitarias (Pytest)
│   ├── test_cli_pipeline.py
│   ├── test_indicators.py
│   ├── test_metrics.py
│   ├── test_parser.py
│   └── test_scanner.py
└── pyproject.toml               # Empaquetado y configuración de herramientas
```

---

## ⚡ Instalación

Requiere **Python 3.10+**.

```bash
git clone https://github.com/fmpalmab/iwannaberich.git
cd iwannaberich

# Instalación base
pip install -e .

# Con dependencias de pruebas
pip install -e ".[dev]"
```

---

## 🚀 Uso Rápido (CLI Unificado)

```bash
# 1. Ejecutar pipeline completo de punta a punta (Consolidar, filtrar, calcular y reportar)
python -m iwannaberich --pipeline

# 2. Escanear oportunidades y anomalías técnicas en el mercado
python -m iwannaberich --scan

# 3. Generar la matriz estocástica completa de riesgo y eficiencia
python -m iwannaberich --matrix

# 4. Imprimir reporte ejecutivo de destacados (Top Sharpe, Defensa, Rebote, Colas gordas)
python -m iwannaberich --report

# 5. Simulación Estocástica de Monte Carlo (Conos de confianza, VaR y CVaR proyectados)
python -m iwannaberich --monte-carlo SQM-B.SN --sims 1000 --horizon 252

# 6. Optimización de Portafolios Markowitz & Frontera Eficiente (Max Sharpe & Min Variance)
python -m iwannaberich --portfolio-opt

# 7. Sincronizar y actualizar cotizaciones a la fecha más reciente (Online Yahoo Finance + Extensión)
python -m iwannaberich --update-data

# 8. Generar gráfico técnico de un activo específico (guarda en PNG de alta resolución)
python -m iwannaberich --plot SQM-B.SN --output data/analisis_SQM.png

# 9. Generar dataset sintético para pruebas locales o CI sin conexión
python -m iwannaberich --synth
```

### Compatibilidad Histórica

Los scripts secuenciales originales en `src/` continúan plenamente operativos delegando al motor moderno:

```bash
python src/01ticker_parser.py
python src/02data_loader.py
python src/03market_processor.py
python src/04scanner.py
python src/05visualizaer.py
python src/06matrix.py
python src/06report.py
```

---

## 🧮 Modelos Cuantitativos y Matemáticos

### 1. Retornos y Ratios de Eficiencia

Se emplean retornos logarítmicos aditivos $r_t = \ln(P_t / P_{t-1})$. Para una tasa libre de riesgo $R_f$:

- **Retorno Anualizado (CAGR aprox):** $\mu_{\text{anual}} = \bar{r} \times 252$
- **Volatilidad Anualizada:** $\sigma_{\text{anual}} = \operatorname{std}(r) \times \sqrt{252}$
- **Sharpe Ratio:** 
  $$\text{Sharpe} = \frac{\mu_{\text{anual}} - R_f}{\sigma_{\text{anual}}}$$
- **Sortino Ratio (penaliza solo desviaciones negativas):**
  $$\text{Sortino} = \frac{\mu_{\text{anual}} - R_f}{\sigma_{\text{downside}} \times \sqrt{252}}, \quad \sigma_{\text{downside}} = \sqrt{\frac{1}{N} \sum_{r_t < 0} r_t^2}$$
- **Calmar Ratio:**
  $$\text{Calmar} = \frac{\mu_{\text{anual}}}{|\operatorname{MaxDrawdown}|}$$

### 2. Gestión de Riesgo de Cola (Extreme Value Theory)

- **Value at Risk (VaR 95% Diario):** Cuantil empírico al 5%:
  $$\operatorname{VaR}_{0.05} = \inf \{ l \in \mathbb{R} : P(r \le l) \ge 0.05 \}$$
- **Conditional VaR (CVaR / Expected Shortfall 95%):** Pérdida esperada en el peor 5% de los escenarios:
  $$\operatorname{CVaR}_{0.05} = \mathbb{E}[r \mid r \le \operatorname{VaR}_{0.05}]$$
- **Curtosis y Asimetría (Skewness):** Detección de activos con colas pesadas (alta probabilidad de cisnes negros) o sesgo positivo de apreciación rápida.

---

## 🧪 Pruebas Automatizadas

```bash
pytest -v
```

La suite cubre parsing regex, indicadores técnicos, propiedades estocásticas de las métricas, señales del escáner y ejecución CLI.

---

## 📜 Licencia

Distribuido bajo la Licencia **MIT**.

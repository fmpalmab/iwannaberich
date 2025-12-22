import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'precios_liquidos.parquet')

def plot_opportunity(ticker, window=252):
    """Grafica Precio vs SMA200 y el Oscilador RSI abajo"""
    
    # Cargar datos
    if not os.path.exists(INPUT_FILE):
        print("Error: No encuentro precios_liquidos.parquet")
        return
    
    df = pd.read_parquet(INPUT_FILE)
    
    if ticker not in df.columns:
        print(f"Ticker {ticker} no encontrado en la base de datos.")
        return
    
    # Preparar datos (último año laboral ~252 días)
    data = df[ticker].tail(window).copy()
    
    # Calcular SMA 200 (necesitamos más historia hacia atrás para que no sea NaN al principio del plot)
    # Truco: Tomamos toda la historia, calculamos y luego cortamos
    full_series = df[ticker]
    sma_200 = full_series.rolling(window=200).mean().tail(window)
    
    # Calcular RSI
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # --- PLOTTEO (Estilo Ingeniería) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Panel 1: Precio
    ax1.plot(data.index, data, label='Precio Cierre', color='black', linewidth=1.5)
    ax1.plot(data.index, sma_200, label='Tendencia (SMA 200)', color='blue', linestyle='--', alpha=0.7)
    ax1.set_title(f"Análisis Técnico: {ticker}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precio (CLP)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Panel 2: RSI
    ax2.plot(rsi.index, rsi, label='RSI (14)', color='purple')
    ax2.axhline(70, color='red', linestyle=':', alpha=0.5) # Sobrecompra
    ax2.axhline(30, color='green', linestyle=':', alpha=0.5) # Sobreventa
    ax2.fill_between(rsi.index, rsi, 30, where=(rsi < 30), color='green', alpha=0.3)
    ax2.fill_between(rsi.index, rsi, 70, where=(rsi > 70), color='red', alpha=0.3)
    ax2.set_ylabel('Oscilador RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Graficamos tus hallazgos del CSV
    print("Graficando AMZNCL.SN (La caída fuerte)...")
    plot_opportunity('AMZNCL.SN') 
    
    print("Graficando PEHUENCHE.SN (La oportunidad en Energía)...")
    plot_opportunity('PEHUENCHE.SN')
    
    print("Graficando PAZ.SN (Inmobiliaria en el suelo)...")
    plot_opportunity('PAZ.SN')
import pandas as pd
import numpy as np
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'precios_liquidos.parquet')

def calculate_rsi(series, period=14):
    """Calcula el Relative Strength Index (Oscilador de momento)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_market():
    print("🛰️  Escaneando el mercado en busca de oportunidades...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ No encuentro {INPUT_FILE}. Corre market_processor.py primero.")
        return

    # Leer datos limpios (Filas=Fechas, Columnas=Tickers)
    df_prices = pd.read_parquet(INPUT_FILE)
    
    results = []

    # Iteramos por cada acción (columna)
    for ticker in df_prices.columns:
        series = df_prices[ticker].dropna()
        
        if len(series) < 200:
            continue # Necesitamos al menos 200 días de historia
            
        current_price = series.iloc[-1]
        
        # --- INDICADORES (Procesamiento de Señal) ---
        
        # 1. Medias Móviles (Filtros de Tendencia)
        sma_50 = series.rolling(window=50).mean().iloc[-1]
        sma_200 = series.rolling(window=200).mean().iloc[-1]
        
        # 2. RSI (Nivel de "Energía" actual)
        # RSI < 30: Sobreventa (Barato) | RSI > 70: Sobrecompra (Caro)
        rsi_current = calculate_rsi(series).iloc[-1]
        
        # 3. Drawdown (Caída desde el máximo anual)
        max_52w = series.tail(252).max()
        drawdown_pct = ((current_price - max_52w) / max_52w) * 100
        
        # 4. Volatilidad (Desviación estándar de los retornos)
        volatility = series.pct_change().std() * np.sqrt(252) * 100 # Anualizada
        
        # --- LOGICA DE CLASIFICACIÓN (El "Algoritmo") ---
        trend = "BAJISTA"
        if current_price > sma_200:
            trend = "ALCISTA"
            
        signal = "NEUTRAL"
        
        # Estrategia 1: "Buy the Dip" (Tendencia Alcista + Recorte temporal)
        if trend == "ALCISTA" and rsi_current < 40:
            signal = "OPORTUNIDAD (DIP)"
            
        # Estrategia 2: "Rebote Táctico" (Acción muy castigada)
        elif rsi_current < 30:
            signal = "SOBREVENTA EXTREMA"
            
        # Estrategia 3: "Momentum" (Rompiendo máximos)
        elif current_price > sma_50 and current_price > sma_200 and rsi_current > 60:
             signal = "FUERZA (MOMENTUM)"

        results.append({
            'Ticker': ticker,
            'Precio': round(current_price, 2),
            'Tendencia (SMA200)': trend,
            'RSI (14)': round(rsi_current, 2),
            'Distancia Max 52W (%)': round(drawdown_pct, 2),
            'Volatilidad Anual (%)': round(volatility, 2),
            'Señal Técnica': signal
        })

    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    # --- FILTROS DE SALIDA ---
    # Mostramos solo las que tienen alguna señal interesante o están en tendencia alcista
    opportunities = df_results[df_results['Señal Técnica'] != "NEUTRAL"].sort_values('RSI (14)')
    
    print(f"\n✅ Análisis completado. Se encontraron {len(opportunities)} candidatos potenciales.\n")
    
    # Mostrar Top Oportunidades
    pd.set_option('display.max_rows', None)
    print(opportunities[['Ticker', 'Precio', 'Señal Técnica', 'RSI (14)', 'Distancia Max 52W (%)']])
    
    # Guardar reporte
    out_path = os.path.join(BASE_DIR, 'data', 'reporte_oportunidades.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\n📄 Reporte completo guardado en: {out_path}")

if __name__ == "__main__":
    analyze_market()
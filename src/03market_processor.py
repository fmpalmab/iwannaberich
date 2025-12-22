import pandas as pd
import os
import numpy as np

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'precios_historicos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_consolidate():
    print("⚙️  Cargando archivos Parquet y consolidando matriz...")
    
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')]
    data_frames = []

    for f in all_files:
        ticker = f.replace('.parquet', '')
        file_path = os.path.join(INPUT_DIR, f)
        
        try:
            df = pd.read_parquet(file_path)
            # Nos quedamos solo con Close y Volume
            df = df[['Close', 'Volume']].copy()
            # Renombramos columnas para que sean únicas por ticker
            df.columns = [f"{ticker}_Close", f"{ticker}_Volume"]
            data_frames.append(df)
        except Exception as e:
            print(f"Error leyendo {ticker}: {e}")

    # Unir todo por el índice (Fecha)
    # concat axis=1 alinea las fechas automáticamente
    market_matrix = pd.concat(data_frames, axis=1)
    
    # Ordenar por fecha
    market_matrix = market_matrix.sort_index()
    
    print(f"✅ Matriz consolidada: {market_matrix.shape[0]} días x {market_matrix.shape[1]} columnas")
    return market_matrix

def filter_by_liquidity(market_matrix, min_daily_volume_clp=5_000_000):
    """
    Filtra acciones que mueven poco dinero.
    Criterio: El promedio del (Precio * Volumen) en los últimos 3 meses debe ser > X CLP.
    """
    print("🔍 Analizando liquidez...")
    
    tickers = [c.split('_')[0] for c in market_matrix.columns if '_Close' in c]
    valid_tickers = []
    
    liquidity_stats = []

    # Usamos los últimos 90 días hábiles (aprox 4-5 meses) para filtrar
    recent_data = market_matrix.tail(90)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        vol_col = f"{ticker}_Volume"
        
        if close_col not in recent_data.columns or vol_col not in recent_data.columns:
            continue
            
        # Dinero transado aprox = Precio Cierre * Volumen
        traded_value = recent_data[close_col] * recent_data[vol_col]
        avg_traded = traded_value.mean()
        
        # Guardamos estadísticas
        liquidity_stats.append({
            'Ticker': ticker,
            'Avg_Traded_CLP': avg_traded,
            'Last_Price': recent_data[close_col].iloc[-1]
        })

        if avg_traded > min_daily_volume_clp:
            valid_tickers.append(ticker)

    df_stats = pd.DataFrame(liquidity_stats).sort_values('Avg_Traded_CLP', ascending=False)
    
    # Guardamos el reporte de liquidez
    df_stats.to_csv(os.path.join(OUTPUT_DIR, 'reporte_liquidez.csv'), index=False)
    
    print(f"📉 Acciones totales: {len(tickers)}")
    print(f"📈 Acciones líquidas (> {min_daily_volume_clp/1e6}M CLP/día): {len(valid_tickers)}")
    print("Top 5 Líquidas:\n", df_stats.head(5))
    
    return valid_tickers

def main():
    # 1. Cargar todo
    full_matrix = load_and_consolidate()
    
    # 2. Filtrar
    liquid_tickers = filter_by_liquidity(full_matrix)
    
    # 3. Guardar solo los datos de precios de las acciones líquidas (limpio)
    # Seleccionamos solo las columnas de _Close de los tickers validos
    close_cols = [f"{t}_Close" for t in liquid_tickers]
    
    clean_prices = full_matrix[close_cols].copy()
    
    # Renombrar columnas quitando el "_Close" para facilitar uso posterior
    clean_prices.columns = liquid_tickers
    
    # Rellenar NaNs:
    # forward fill (si no hubo precio hoy, asumo el de ayer)
    clean_prices = clean_prices.ffill()
    
    output_file = os.path.join(OUTPUT_DIR, 'precios_liquidos.parquet')
    clean_prices.to_parquet(output_file)
    print(f"💾 Archivo maestro guardado: {output_file}")

if __name__ == "__main__":
    main()
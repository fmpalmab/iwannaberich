import yfinance as yf
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
PRICES_DIR = os.path.join(DATA_DIR, 'precios_historicos')
os.makedirs(PRICES_DIR, exist_ok=True)

def download_from_master():
    master_path = os.path.join(DATA_DIR, 'maestro_tickers.csv')
    
    if not os.path.exists(master_path):
        print("Corre primero ticker_parser.py")
        return

    df_master = pd.read_csv(master_path)
    tickers = df_master['Ticker'].tolist()
    
    print(f"⬇️ Descargando historia para {len(tickers)} activos seleccionados...")
    
    # Descarga masiva optimizada
    # Periodo 'max' nos da toda la historia disponible
    data = yf.download(tickers, period="5y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    
    # Guardado inteligente
    success = 0
    for ticker in tickers:
        try:
            df_t = data[ticker].copy()
            # Filtro de Ingeniero: Si está vacío o lleno de NaNs, ignorar
            if df_t.empty or df_t['Close'].isna().all():
                continue
                
            # Guardamos en Parquet (rápido y liviano)
            filename = os.path.join(PRICES_DIR, f"{ticker}.parquet")
            df_t.to_parquet(filename)
            success += 1
        except Exception:
            pass # Si falla uno, seguimos con el siguiente
            
    print(f"✅ Proceso terminado. {success} archivos guardados en {PRICES_DIR}")

if __name__ == "__main__":
    download_from_master()
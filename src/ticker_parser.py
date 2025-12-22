import re
import pandas as pd
import os

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
RAW_FILE_PATH = os.path.join(DATA_DIR, 'raw_tickers.txt')

def parse_tickers():
    print("🧹 Iniciando limpieza PROFUNDA de tickers (V3.0)...")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ Error: No encuentro {RAW_FILE_PATH}. Asegúrate de que tenga la lista sucia.")
        return []

    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # PASO 1: Encontrar todo lo que termine en .SN (capturando los números previos también)
    # \d* captura los números pegados al inicio (ej: 275LLTM.SN)
    # [A-Z0-9-]+ captura las letras del ticker
    raw_matches = re.findall(r"(\d*[A-Z0-9-]+\.SN)", content)
    
    clean_tickers = []
    
    print(f"🔍 Analizando {len(raw_matches)} posibles tickers...")

    for item in raw_matches:
        # Etapa 1: Eliminar números iniciales (ej: '275LLTM.SN' -> 'LLTM.SN')
        # Usamos regex para sustituir dígitos al inicio (^) por nada
        no_digits = re.sub(r"^\d+", "", item)
        
        # Etapa 2: Corregir la letra duplicada (ej: 'LLTM.SN' -> 'LTM.SN')
        # La fuente de datos parece haber duplicado siempre la primera letra.
        # Lógica: Si la primera letra es igual a la segunda, cortamos la primera.
        final_ticker = no_digits
        if len(final_ticker) > 2 and final_ticker[0] == final_ticker[1]:
            final_ticker = final_ticker[1:]
            
            # Etapa 2.1: Caso especial "Triple Letra" (ej: AAAPL -> AAPL)
            # Si después de cortar, sigue habiendo duplicado al inicio y NO es un ticker real conocido
            # (En tu lista AAAPL.SN era AAPL.SN, por ende había 3 A's al inicio: 12AAAPL)
            if len(final_ticker) > 2 and final_ticker[0] == final_ticker[1]:
                 # Excepción: CCU.SN es real. No cortamos si es CCU.
                 if final_ticker != "CCU.SN":
                     final_ticker = final_ticker[1:]

        # Filtro de basura
        if "Símbolo" in final_ticker: continue
        
        clean_tickers.append(final_ticker)

    # Eliminar duplicados y ordenar
    unique_tickers = sorted(list(set(clean_tickers)))
    
    print(f"✅ Limpieza completada. {len(unique_tickers)} tickers válidos recuperados.")
    print(f"Ejemplo corrección: '{raw_matches[0]}' -> '{unique_tickers[0]}'")

    # --- CLASIFICACIÓN Y GUARDADO ---
    classified = []
    for t in unique_tickers:
        category = "CHILE_EQUITY"
        # Clasificación simple
        if "CFI" in t or "-E.SN" in t: category = "FONDO_INVERSION"
        elif t in ["AAPL.SN", "MSFT.SN", "TSLA.SN", "GOOGL.SN", "NVDA.SN", "AMZN.SN"]: category = "US_TECH_ENERGY_CLP"
        
        classified.append({'Ticker': t, 'Category': category})
    
    df = pd.DataFrame(classified)
    output_path = os.path.join(DATA_DIR, 'maestro_tickers.csv')
    df.to_csv(output_path, index=False)
    print(f"📁 Archivo maestro sobrescrito en: {output_path}")

if __name__ == "__main__":
    parse_tickers()
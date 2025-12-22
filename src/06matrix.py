import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis, jarque_bera

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'precios_historicos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_FREE_RATE = 0.045 # Tasa libre de riesgo referencial (Bono Tesoro / Depósito Plazo ~4.5%)

def load_data():
    print(f"📂 Cargando historial desde: {INPUT_DIR}")
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')]
    
    if not files:
        print("❌ No se encontraron archivos .parquet. Ejecuta el data_loader primero.")
        return None

    data_frames = []
    for f in files:
        try:
            path = os.path.join(INPUT_DIR, f)
            df = pd.read_parquet(path)
            # Nos aseguramos de tener solo el precio de cierre y renombrarlo
            ticker = f.replace('.parquet', '')
            if 'Close' in df.columns:
                series = df['Close']
                series.name = ticker
                data_frames.append(series)
        except Exception as e:
            print(f"⚠️ Error leyendo {f}: {e}")

    print(f"✅ Se cargaron {len(data_frames)} activos. Consolidando matriz...")
    # Unir todo en un solo DataFrame alineado por fecha
    prices = pd.concat(data_frames, axis=1).sort_index()
    return prices

def calculate_metrics(prices):
    print("🧮 Calculando métricas estocásticas y financieras...")
    
    # 1. Preparar Retornos (Log returns para propiedades estadísticas aditivas)
    returns = np.log(prices / prices.shift(1))
    
    # DataFrame de resultados
    metrics = pd.DataFrame(index=returns.columns)
    
    # --- A. RETORNO Y TENDENCIA ---
    # CAGR (Compound Annual Growth Rate) aproximado
    metrics['Retorno_Anualizado'] = returns.mean() * 252
    # Probabilidad de días verdes
    metrics['Prob_Subida'] = returns.apply(lambda x: (x > 0).mean())
    
    # --- B. RIESGO (VOLATILIDAD) ---
    metrics['Volatilidad_Anual'] = returns.std() * np.sqrt(252)
    metrics['Desv_Bajista'] = returns.apply(lambda x: x[x < 0].std()) * np.sqrt(252) # Downside Deviation
    
    # --- C. RATIOS DE EFICIENCIA (Ingeniería Financiera) ---
    # Sharpe: (Retorno - Rf) / Volatilidad
    metrics['Sharpe_Ratio'] = (metrics['Retorno_Anualizado'] - RISK_FREE_RATE) / metrics['Volatilidad_Anual']
    # Sortino: (Retorno - Rf) / Volatilidad Bajista (Penaliza solo el riesgo malo)
    metrics['Sortino_Ratio'] = (metrics['Retorno_Anualizado'] - RISK_FREE_RATE) / metrics['Desv_Bajista']
    
    # --- D. DISTRIBUCIÓN ESTADÍSTICA (Momentos) ---
    # Skewness (Sesgo): Negativo = Golpes bajistas fuertes frecuentes
    metrics['Skewness'] = returns.apply(lambda x: skew(x.dropna()))
    # Kurtosis (Curtosis): > 3 indica "Colas Gordas" (Eventos cisne negro probables)
    metrics['Kurtosis'] = returns.apply(lambda x: kurtosis(x.dropna(), fisher=False))
    
    # --- E. RIESGO DE COLA (Risk Management) ---
    # VaR (Value at Risk) 95%: Lo máximo que esperas perder en un día malo (con 95% confianza)
    metrics['VaR_95_Diario'] = returns.apply(lambda x: np.percentile(x.dropna(), 5))
    # CVaR (Conditional VaR): El promedio de pérdidas cuando se rompe el VaR (el escenario desastre)
    def calculate_cvar(x, p=5):
        data = x.dropna()
        threshold = np.percentile(data, p)
        return data[data <= threshold].mean()
    metrics['CVaR_95_Diario'] = returns.apply(calculate_cvar)
    
    # --- F. DRAWDOWN (Caída Máxima) ---
    def max_drawdown(series):
        comp_ret = (1 + series).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()
    
    # Usamos retornos simples para drawdown exacto
    simple_ret = prices.pct_change()
    metrics['Max_Drawdown'] = simple_ret.apply(max_drawdown)
    
    # Calmar Ratio: Retorno / Max Drawdown (Cuánto gano por el dolor de la peor caída)
    metrics['Calmar_Ratio'] = metrics['Retorno_Anualizado'] / abs(metrics['Max_Drawdown'])

    return metrics

def main():
    prices = load_data()
    if prices is not None:
        # Filtrar activos con muy poca historia (< 6 meses)
        prices = prices.dropna(thresh=120, axis=1) 
        
        full_matrix = calculate_metrics(prices)
        
        # Limpieza final (infinitos o NaNs por divisiones cero)
        full_matrix = full_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        
        output_path = os.path.join(OUTPUT_DIR, 'matriz_estadistica_completa.csv')
        full_matrix.to_csv(output_path)
        print(f"\n💾 Matriz guardada exitosamente en: {output_path}")
        print(f"   Dimensiones: {full_matrix.shape[0]} activos x {full_matrix.shape[1]} métricas")

if __name__ == "__main__":
    main()
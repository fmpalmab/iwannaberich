import pandas as pd
import os

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'matriz_estadistica_completa.csv')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'data', 'reporte_destacados.txt')

def generate_report():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ No se encuentra {INPUT_FILE}. Ejecuta statistical_matrix.py primero.")
        return

    df = pd.read_csv(INPUT_FILE, index_col=0)
    print(f"📊 Analizando {len(df)} activos...")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=== REPORTE DE INTELIGENCIA DE MERCADO ===\n")
        f.write(f"Total Activos Analizados: {len(df)}\n")
        f.write("=========================================\n\n")

        # 1. LOS REYES DE LA EFICIENCIA (Sharpe Ratio)
        # Activos que pagan mejor por unidad de riesgo asumido.
        top_sharpe = df.sort_values('Sharpe_Ratio', ascending=False).head(10)
        f.write("🏆 TOP 10 EFICIENCIA (Sharpe Ratio)\n")
        f.write("(Mejor retorno ajustado por riesgo)\n")
        f.write(top_sharpe[['Sharpe_Ratio', 'Retorno_Anualizado', 'Volatilidad_Anual']].to_string())
        f.write("\n\n" + "-"*40 + "\n\n")

        # 2. LOS "SEGUROS" (Menor Volatilidad)
        # Para momentos de incertidumbre.
        low_vol = df.sort_values('Volatilidad_Anual', ascending=True).head(10)
        f.write("🛡️ TOP 10 DEFENSA (Menor Volatilidad)\n")
        f.write("(Activos más estables)\n")
        f.write(low_vol[['Volatilidad_Anual', 'Max_Drawdown', 'VaR_95_Diario']].to_string())
        f.write("\n\n" + "-"*40 + "\n\n")

        # 3. LAS OPORTUNIDADES DE "REBOTE" (Deep Value / Calmar)
        # Activos con buen retorno histórico pero que han tenido caídas fuertes (Drawdown).
        # Filtramos primero que tengan retorno positivo.
        value_play = df[df['Retorno_Anualizado'] > 0].sort_values('Max_Drawdown', ascending=True).head(10)
        f.write("📉 TOP 10 CAÍDAS FUERTES (Potencial Rebote)\n")
        f.write("(Activos rentables con mayor descuento desde máximos)\n")
        f.write(value_play[['Max_Drawdown', 'Retorno_Anualizado', 'Calmar_Ratio']].to_string())
        f.write("\n\n" + "-"*40 + "\n\n")

        # 4. RIESGO DE COLA (Cisnes Negros)
        # Kurtosis alta = Probabilidad de movimientos extremos inesperados.
        fat_tails = df.sort_values('Kurtosis', ascending=False).head(10)
        f.write("⚠️ ALERTA DE RIESGO: COLAS GORDAS (Kurtosis Alta)\n")
        f.write("(Mayor probabilidad de eventos extremos/cisnes negros)\n")
        f.write(fat_tails[['Kurtosis', 'Skewness', 'CVaR_95_Diario']].to_string())
        f.write("\n\n" + "-"*40 + "\n\n")
        
        # 5. ASIMETRÍA POSITIVA (Billete de Lotería)
        # Skewness > 0: Es más probable que suba fuerte a que baje fuerte.
        pos_skew = df.sort_values('Skewness', ascending=False).head(10)
        f.write("🚀 ASIMETRÍA POSITIVA (Skewness > 0)\n")
        f.write("(Históricamente, sus subidas son más violentas que sus bajadas)\n")
        f.write(pos_skew[['Skewness', 'Retorno_Anualizado', 'Prob_Subida']].to_string())
        f.write("\n\n")

    print(f"✅ Reporte generado: {OUTPUT_REPORT}")
    
    # Imprimir un adelanto en consola
    print("\n--- ADELANTO: TOP 5 SHARPE RATIO ---")
    print(top_sharpe[['Sharpe_Ratio', 'Retorno_Anualizado', 'Volatilidad_Anual']])

if __name__ == "__main__":
    generate_report()
"""
Compatibility wrapper for 05visualizaer.py.
Forwards to iwannaberich.visualizer.
"""

from iwannaberich.visualizer import plot_technical_analysis

def main():
    print("Generando gráficos de análisis técnico...")
    plot_technical_analysis("AMZNCL.SN", output_path="analisis_AMZNCL.png")
    plot_technical_analysis("PEHUENCHE.SN", output_path="analisis_PEHUENCHE.png")
    plot_technical_analysis("PAZ.SN", output_path="analisis_PAZ.png")
    print("Gráficos generados exitosamente.")

if __name__ == "__main__":
    main()
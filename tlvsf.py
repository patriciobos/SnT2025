import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from matplotlib.ticker import AutoMinorLocator

# === Configuración general ===
carpeta_output = Path("output-data")
carpeta_figuras = Path("figuras")
carpeta_figuras.mkdir(parents=True, exist_ok=True)

# === Función para graficar un archivo CSV ===
def graficar_csv(archivo):
    try:
        nombre_base = archivo.stem  # sin extensión .csv
        #nombre_base = archivo.stem.replace("TL_", "")
        nombre_figura = f"{nombre_base}.png"
        
        df = pd.read_csv(archivo)

        columnas_tl = [col for col in df.columns if col.startswith("tl_")]
        if "frecuencias" not in df.columns or not columnas_tl:
            print(f"Omitiendo archivo sin formato esperado: {archivo}")
            return

        plt.figure()
        for col in columnas_tl:
            plt.plot(df["frecuencias"], df[col], marker='o', label=col)

        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("TL (dB)")
        plt.title(f"Pérdidas por Transmisión - {nombre_base}")
        
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        
        plt.grid(True, which='major', linestyle='-', linewidth=0.75)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(carpeta_figuras)/nombre_figura, dpi=300, transparent=False)
        plt.close()
    
    except Exception as e:
        print(f"Error al procesar {archivo}: {e}")

# === Ejecutar en paralelo ===
if __name__ == "__main__":
    archivos_csv = sorted(carpeta_output.glob("*.csv"))

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(graficar_csv, archivos_csv)

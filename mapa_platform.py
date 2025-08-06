import os
import re
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np

# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "zais"               # opciones: 'zais', 'gsj', 'arasj'
VAR_TL = "tl_z_8"           # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"

# ==== FUNCIONES AUXILIARES ====

def extraer_frecuencia(nombre_archivo):
    match = re.search(r"f(\d+(\.\d+)?)\s*Hz", nombre_archivo)
    return float(match.group(1)) if match else None

def corregir_tl_column(col):
    col_corr = []
    correcciones = 0

    for val in col:
        try:
            fval = float(val)
            if fval <= 2000:
                col_corr.append(fval)
                continue
        except:
            pass

        val_str = str(val).strip()
        val_digits = ''.join(filter(str.isdigit, val_str))

        if len(val_digits) >= 4:
            corrected = float(val_digits[:3] + '.' + val_digits[3:])
            col_corr.append(corrected)
            correcciones += 1
        else:
            col_corr.append(np.nan)

    return pd.Series(col_corr), correcciones


def procesar_archivo(ruta_archivo):
    try:
        nombre_archivo = os.path.basename(ruta_archivo)
        if ZONA not in nombre_archivo:
            return

        frecuencia = extraer_frecuencia(nombre_archivo)
        if frecuencia is None:
            print(f"[WARN] No se pudo extraer la frecuencia de {nombre_archivo}")
            return

        df = pd.read_csv(ruta_archivo)

       
        #df[VAR_TL], cant_corregidas = corregir_tl_column(df[VAR_TL])
        #print(f"[INFO] {os.path.basename(ruta_archivo)}: {cant_corregidas} valores corregidos en la columna {VAR_TL}")

        #df = df.dropna(subset=[VAR_TL])  # eliminar filas que no pudieron corregirse

        columnas_necesarias = {'lat', 'lon', VAR_TL}
        if not columnas_necesarias.issubset(df.columns):
            print(f"[ERROR] Columnas faltantes en {nombre_archivo}")
            return

        lat = df['lat']
        lon = df['lon']
        tl = df[VAR_TL]

        # === GRAFICAR MAPA ===
        fig, ax = plt.subplots(figsize=(10, 8))
        m = Basemap(projection='merc',
                    llcrnrlat=lat.min() - 2, urcrnrlat=lat.max() + 2,
                    llcrnrlon=lon.min() - 2, urcrnrlon=lon.max() + 2,
                    resolution='i', ax=ax)

        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90., 0., 2.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-70., -50., 2.), labels=[0, 0, 0, 1])

        x, y = m(df['lon'].values, df['lat'].values)

        # Máscara para TL < 210
        mask_low = tl < 210
        mask_high = ~mask_low

        # Puntos con TL >= 210 (coloreados por escala)
        sc = m.scatter(np.array(x)[mask_high], np.array(y)[mask_high],
                    c=tl[mask_high], cmap='viridis', marker='o',
                    edgecolor='k', s=50, label=f"{VAR_TL} ≥ 210")

        # Puntos con TL < 210 en rojo
        m.scatter(np.array(x)[mask_low], np.array(y)[mask_low],
                color='red', marker='x', s=60, label=f"{VAR_TL} < 210")

        # Leyenda y colorbar
        cbar = m.colorbar(sc, location='right', pad="5%")
        cbar.set_label(f"{VAR_TL} [dB]")
        plt.legend(loc='lower left')

        cbar = m.colorbar(sc, location='right', pad="5%")
        cbar.set_label(f"{VAR_TL} [dB]")

        plt.title(f"Zona: {ZONA.upper()} - Frecuencia: {frecuencia:.1f} Hz - {VAR_TL}")

        os.makedirs(CARPETA_OUTPUT, exist_ok=True)
        output_path = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{frecuencia:.1f}Hz_{VAR_TL}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Procesado {nombre_archivo} → {output_path}")

    except Exception as e:
        print(f"[ERROR] Al procesar {ruta_archivo}: {e}")

# ==== MAIN: PROCESAR EN PARALELO ====

if __name__ == "__main__":
    archivos = [os.path.join(CARPETA_INPUT, f)
                for f in os.listdir(CARPETA_INPUT)
                if f.endswith(".csv") and ZONA in f]

    print(f"Procesando {len(archivos)} archivos para la zona '{ZONA}' usando {multiprocessing.cpu_count()} núcleos...")

    with multiprocessing.Pool() as pool:
        pool.map(procesar_archivo, archivos)

    print("✅ Todos los archivos procesados.")

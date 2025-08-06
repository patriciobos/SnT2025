import os
import re
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np

from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as MplPolygon
import alphashape



# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "gsj"               # opciones: 'zais', 'gsj', 'arasj'
VAR_TL = "tl_z_8"           # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"

# ==== FUNCIONES AUXILIARES ====

def extraer_frecuencia(nombre_archivo):
    match = re.search(r"f(\d+(\.\d+)?)\s*Hz", nombre_archivo)
    return float(match.group(1)) if match else None

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

        # # === Calcular el polígono convexo que encierra los puntos ===
        # puntos = list(zip(x, y))
        # multipoint = MultiPoint(puntos)
        # if not multipoint.is_empty and len(multipoint.geoms) > 2:
        #     convex_hull = multipoint.convex_hull

        #     # Asegurar que sea un polígono (puede ser una línea si hay pocos puntos)
        #     if convex_hull.geom_type == 'Polygon':
        #         x_hull, y_hull = convex_hull.exterior.xy
        #         hull_polygon = MplPolygon(
        #             list(zip(x_hull, y_hull)),
        #             closed=True,
        #             fill=False,
        #             edgecolor='orange',
        #             linewidth=2,
        #             label='Área de datos'
        #         )
        #         ax.add_patch(hull_polygon)

        # === Calcular el polígono cóncavo (alpha shape) sobre coordenadas geográficas ===
        geo_points = list(zip(df['lon'], df['lat']))  # Usar lon/lat reales para el alpha shape
        if len(geo_points) >= 4:
            alpha_shape = alphashape.alphashape(geo_points, alpha=.5)
            if alpha_shape.geom_type == 'Polygon':
                lon_alpha, lat_alpha = alpha_shape.exterior.xy
                x_alpha, y_alpha = m(lon_alpha, lat_alpha)  # Proyectar los puntos del borde
                alpha_polygon = MplPolygon(
                    list(zip(x_alpha, y_alpha)),
                    closed=True,
                    fill=False,
                    edgecolor='blue',
                    linewidth=2.5,
                    label='Área de datos'
                )
                ax.add_patch(alpha_polygon)

                
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

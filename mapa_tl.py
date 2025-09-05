import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# === Configuración general ===
carpeta_input = Path("input-data")
carpeta_figuras = Path("figuras")
carpeta_figuras.mkdir(exist_ok=True)

map_config = {
    "projection": "merc",
    "llcrnrlat": -47,
    "urcrnrlat": -38,
    "llcrnrlon": -66,
    "urcrnrlon": -55,
    "resolution": "i"
}

# === Función para verificar si puntos están dentro del convex hull ===
def points_in_hull(xy_points, hull_points):
    hull = Delaunay(hull_points)
    return hull.find_simplex(xy_points) >= 0

# === Función para graficar ===
def graficar_mapa_tl8(archivo_csv):
    try:
        df = pd.read_csv(archivo_csv)
        if not {"lat", "lon", "tl_z_8"}.issubset(df.columns):
            print(f"Saltando {archivo_csv.name}: columnas requeridas no presentes.")
            return

        lats = df["lat"].values
        lons = df["lon"].values
        values = df["tl_z_8"].values

        m = Basemap(**map_config)
        x, y = m(lons, lats)

        # Grilla para interpolación
        xi = np.linspace(min(x), max(x), 300)
        yi = np.linspace(min(y), max(y), 300)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), values, (xi, yi), method="linear")

        # Máscara con convex hull
        puntos = np.column_stack((x, y))
        hull = ConvexHull(puntos)
        inside = points_in_hull(np.column_stack((xi.flatten(), yi.flatten())), puntos[hull.vertices])
        mask = inside.reshape(xi.shape)

        # Extraer frecuencia del nombre
        match = re.search(r"f(\d+)\s*Hz", archivo_csv.stem)
        freq_label = f"{match.group(1)} Hz" if match else "desconocida"
        nombre_figura = f"mapa_TL_z_8_f{match.group(1)}Hz.png" if match else f"mapa_TL_z_8_{archivo_csv.stem}.png"

        # === Figura ===
        fig, ax = plt.subplots(figsize=(8, 6))
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90, 90, 2), labels=[1,0,0,0], linewidth=0.2)
        m.drawmeridians(np.arange(-180, 180, 2), labels=[0,0,0,1], linewidth=0.2)

        # Plot con máscara (alpha=0 fuera del convex hull)
        cs = m.contourf(xi, yi, zi, cmap="viridis", levels=40, alpha=1.0)
        cs.collections[0].set_alpha(1.0)
        zi_masked = np.ma.masked_where(~mask, zi)
        m.contourf(xi, yi, zi_masked, cmap="viridis", levels=40)

        plt.title(f"TL_z_8 @ {freq_label}")
        plt.colorbar(cs, label="TL [dB]", shrink=0.7)
        fig.savefig(carpeta_figuras / nombre_figura, dpi=300)
        plt.close(fig)

        print(f"Guardado: {carpeta_figuras / nombre_figura}")

    except Exception as e:
        print(f"Error al procesar {archivo_csv.name}: {e}")

# === Ejecutar en paralelo ===
if __name__ == "__main__":
    archivos_csv = sorted(carpeta_input.glob("*.csv"))
    if not archivos_csv:
        print("No se encontraron archivos CSV en la carpeta de entrada.")
        exit(1)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(graficar_mapa_tl8, archivos_csv)

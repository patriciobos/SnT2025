import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm as mpl_cm
from shapely.geometry import Point
import alphashape
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import multiprocessing as mp
import re

# Configuración
carpeta_datos = "input-data"
carpeta_figuras = "figuras"
TLmin = 80
TLmax = 220

col_lat = 'lat'
col_lon = 'lon'
col_TL_default = 'tl_z_8'

# Bounding box fijo
lon_min, lon_max = -70, -45
lat_min, lat_max = -55, -35

# Lista de ciudades
ciudades_argentinas = [
    {"nombre": "Mar del Plata", "lat": -38.0023, "lon": -57.5575},
    {"nombre": "Bahía Blanca", "lat": -38.7196, "lon": -62.2724},
    {"nombre": "Puerto Madryn", "lat": -42.7692, "lon": -65.0385},
    {"nombre": "Trelew", "lat": -43.2489, "lon": -65.3051},
    {"nombre": "Comodoro Rivadavia", "lat": -45.8647, "lon": -67.4822},
    {"nombre": "Río Gallegos", "lat": -51.6230, "lon": -69.2168},
]

# Crear carpeta si no existe
os.makedirs(carpeta_figuras, exist_ok=True)

def extraer_frecuencia(nombre_archivo):
    match = re.search(r"-f([0-9.]+)\s*Hz", nombre_archivo)
    return f"{match.group(1)} Hz" if match else os.path.splitext(nombre_archivo)[0]

def procesar_archivo(args):
    archivo, col_TL = args
    print(f"Procesando: {archivo}")

    df = pd.read_csv(archivo)
    if col_TL not in df.columns:
        print(f"Columna {col_TL} no encontrada en {archivo}")
        return

    df[col_TL] = pd.to_numeric(df[col_TL], errors='coerce')
    df[col_lat] = pd.to_numeric(df[col_lat], errors='coerce')
    df[col_lon] = pd.to_numeric(df[col_lon], errors='coerce')

    # Filtrar datos   
    df_filtrado = df.dropna(subset=[col_TL, col_lat, col_lon])
    df_filtrado = df_filtrado[df_filtrado[col_TL].between(TLmin, TLmax)]

    if df_filtrado.empty:
        print(f"No hay datos válidos en {archivo}")
        return

    lat_f = df_filtrado[col_lat].values
    lon_f = df_filtrado[col_lon].values
    TL = df_filtrado[col_TL].values

    # Grilla
    grid_lat = np.linspace(lat_min, lat_max, 300)
    grid_lon = np.linspace(lon_min, lon_max, 300)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Interpolación
    points = np.column_stack((lon_f, lat_f))
    grid_TL = griddata(points, TL, (grid_lon2d, grid_lat2d), method='linear')

    # Alpha shape
    alpha = 3.0
    concave_hull = alphashape.alphashape(points, alpha)
    mask = np.array([concave_hull.contains(Point(x, y)) for x, y in zip(grid_lon2d.ravel(), grid_lat2d.ravel())])
    grid_TL_masked = np.full_like(grid_TL, np.nan)
    grid_TL_masked.ravel()[mask] = grid_TL.ravel()[mask]

    # Basemap
    fig = plt.figure(figsize=(10, 9))
    ax_main = fig.add_axes((0.1, 0.1, 0.85, 0.85))

    m = Basemap(projection='merc',
                llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max,
                resolution='i', ax=ax_main)

    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawparallels(range(lat_min, lat_max + 1, 5), labels=[1,0,0,0])
    m.drawmeridians(range(lon_min, lon_max + 1, 5), labels=[0,0,0,1])

    x, y = m(grid_lon2d, grid_lat2d)
    norm = Normalize(vmin=TLmin, vmax=TLmax)
    #norm = Normalize(vmin=np.nanmin(grid_TL_masked), vmax=np.nanmax(grid_TL_masked))
    cmap = mpl_cm.get_cmap('viridis')
    im = m.pcolormesh(x, y, grid_TL_masked, cmap=cmap, norm=norm, shading='auto')

    cbar = plt.colorbar(im, ax=ax_main, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"{col_TL} (dB)")

    # Ciudades
    for ciudad in ciudades_argentinas:
        cx, cy = m(ciudad["lon"], ciudad["lat"])
        m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
        plt.text(cx + 5000, cy + 5000, ciudad["nombre"], fontsize=8, ha='center', va='bottom')

    # Rectángulo ZAIS
    zais_bounds = [[-38.5833, -57.3333], [-38.0, -55.55]]
    lats_zais = [zais_bounds[0][0], zais_bounds[1][0]]
    lons_zais = [zais_bounds[0][1], zais_bounds[1][1]]
    x1, y1 = m(lons_zais[0], lats_zais[0])
    x2, y2 = m(lons_zais[1], lats_zais[1])
    m.plot([x1, x2, x2, x1, x1],
           [y1, y1, y2, y2, y1],
           color='black', linestyle='--', linewidth=1.5)
    plt.text((x1 + x2) / 2, y1 - 100000, "ZAIS", ha='center', va='top', fontsize=9)

    # Título
    frecuencia_str = extraer_frecuencia(os.path.basename(archivo))
    ax_main.set_title(f"Mapa de {col_TL} - {frecuencia_str}", fontsize=14)

    # Guardar imagen
    nombre_base = os.path.splitext(os.path.basename(archivo))[0]
    salida = os.path.join(carpeta_figuras, f"{nombre_base}_{col_TL}.png")
    fig.savefig(salida, dpi=300, bbox_inches='tight')
    fig.tight_layout()
    plt.close()
    print(f"Guardado: {salida}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_TL', type=str, default=col_TL_default,
                        help="Nombre de la columna de TL a graficar (ej: tl_z_8, tl_half_z, tl_z_max)")
    args = parser.parse_args()

    archivos = glob.glob(os.path.join(carpeta_datos, "*.csv"))
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(procesar_archivo, [(archivo, args.col_TL) for archivo in archivos])

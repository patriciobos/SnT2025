import os
import re
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, Point
from matplotlib.patches import Polygon as MplPolygon
import alphashape
from scipy.interpolate import griddata
import geopandas as gpd
import imageio



# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "arasj"               # opciones: 'zais', 'gsj', 'arasj'
VAR_TL = "tl_z_8"           # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"
UMBRAL_TL = 200
ALPHA = 0.35
MASCARA_SHP = "Capas/plataforma_continental/plataforma_continentalPolygon.shp"
FILTRO_TL_MIN = 30
MERIDIANO_CORTE = -56
PARALELO_SUR = -56
PARALELO_NORTE = -22
PARALELO_INTERPOLACION = -55

# Coordenadas objetivo con campo opcional "nombre"
coordenadas_objetivo = [
    {"lat": -38.5092, "lon": -56.4850, "nombre": "ZAIS"},
    {"lat": -44.9512, "lon": -63.8894, "nombre": "GSJ"},
    {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},
]

# ==== FUNCIONES AUXILIARES ====
def extraer_frecuencia(nombre_archivo):
    match = re.search(r"f(\d+(\.\d+)?)\s*Hz", nombre_archivo)
    return float(match.group(1)) if match else None

def obtener_punto_zona(zona):
    for entry in coordenadas_objetivo:
        if entry["nombre"].lower() == zona.lower():
            return entry["lon"], entry["lat"], entry["nombre"]
    return None, None, None

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
        df = df[df[VAR_TL] > FILTRO_TL_MIN]

        columnas_necesarias = {'lat', 'lon', VAR_TL, 'bat'}
        if not columnas_necesarias.issubset(df.columns):
            print(f"[ERROR] Columnas faltantes en {nombre_archivo}")
            return

        # === GRAFICAR MAPA ===
        fig, ax = plt.subplots(figsize=(10, 8))
        m = Basemap(projection='merc',
                    llcrnrlat=PARALELO_SUR,
                    urcrnrlat=PARALELO_NORTE,
                    llcrnrlon=df['lon'].min() - 2,
                    urcrnrlon=df['lon'].max() + 2,
                    resolution='i', ax=ax)
        
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawparallels(np.arange(-90., 0., 2.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-70., -50., 2.), labels=[0, 0, 0, 1])
        m.drawcoastlines()
        m.drawcountries()
        

        # === POLÍGONO CÓNCAVO ===
        alpha_shape = alphashape.alphashape(list(zip(df['lon'], df['lat'])), alpha=ALPHA)

        # === MALLA E INTERPOLACIÓN ===
        lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 500)
        lat_grid = np.linspace(PARALELO_INTERPOLACION, PARALELO_NORTE, 500)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.c_[lon_mesh.ravel(), lat_mesh.ravel()]

        # === MÁSCARA SHP ===
        gdf_mascara = gpd.read_file(MASCARA_SHP).to_crs("EPSG:4326")

        in_alpha = np.array([alpha_shape.contains(Point(lon, lat)) for lon, lat in grid_points])
        points_in_alpha = grid_points[in_alpha & (grid_points[:, 0] > MERIDIANO_CORTE)]

        gdf_grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), crs="EPSG:4326")
        gdf_grid_left = gdf_grid[grid_points[:, 0] <= MERIDIANO_CORTE]
        gdf_grid_left = gpd.sjoin(gdf_grid_left, gdf_mascara, predicate="intersects", how="inner")
        points_in_mask = np.array([[pt.x, pt.y] for pt in gdf_grid_left.geometry])

        all_valid_points = np.vstack([points_in_alpha, points_in_mask])

        # === INTERPOLACIÓN ===
        tl_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df[VAR_TL], xi=all_valid_points, method='linear')
        bat_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df['bat'], xi=all_valid_points, method='linear')

        mask_valid = (tl_interp < UMBRAL_TL) & (bat_interp > 30)
        final_points = all_valid_points[mask_valid]
        final_tl = tl_interp[mask_valid]

        # === GRAFICAR PUNTOS INTERPOLADOS ===
        x_final, y_final = m(final_points[:, 0], final_points[:, 1])
        sc_interp = m.scatter(x_final, y_final, c=final_tl, cmap='viridis', vmin=FILTRO_TL_MIN, vmax=UMBRAL_TL,
                              marker='s', s=20, edgecolor='none')

        # === GRAFICAR CONTORNO SHP ===
        for geom in gdf_mascara.geometry:
            if geom.geom_type == 'Polygon':
                x_mask, y_mask = m(*geom.exterior.xy)
                ax.plot(x_mask, y_mask, color='gray', linewidth=1.5)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x_mask, y_mask = m(*poly.exterior.xy)
                    ax.plot(x_mask, y_mask, color='gray', linewidth=1.5)

        # === GRAFICAR PUNTO DE CÁLCULO ===
        punto_lon, punto_lat, punto_nombre = obtener_punto_zona(ZONA)
        if punto_lon is not None:
            x_punto, y_punto = m(punto_lon, punto_lat)
            m.plot(x_punto, y_punto, 'r*', markersize=8)
            plt.text(x_punto + 5000, y_punto + 5000, punto_nombre, fontsize=10, color='red', weight='bold')
        
        # Inlet planisferio
        ax_inlet = fig.add_axes([0.55, 0.1, 0.22, 0.22])
        m_inlet = Basemap(projection='cyl', resolution='c', ax=ax_inlet)
        m_inlet.drawcoastlines(linewidth=0.5)
        m_inlet.drawcountries(linewidth=0.5)
        m_inlet.drawmapboundary(fill_color='lightblue')
        m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')
        
        # === TÍTULO Y GUARDADO ===
        cbar = m.colorbar(sc_interp, location='right', pad="5%")
        cbar.set_label(f"{VAR_TL} [dB]")
        #plt.legend(loc='lower left')
        plt.title(f"Zona: {ZONA.upper()} - Frecuencia: {frecuencia:.1f} Hz - {VAR_TL}")

        os.makedirs(CARPETA_OUTPUT, exist_ok=True)
        output_path = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{frecuencia:.1f}Hz_{VAR_TL}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Procesado {nombre_archivo} → {output_path}")

    except Exception as e:
        print(f"[ERROR] Al procesar {ruta_archivo}: {e}")

# ==== MAIN ====
if __name__ == "__main__":
    archivos = [os.path.join(CARPETA_INPUT, f)
                for f in os.listdir(CARPETA_INPUT)
                if f.endswith(".csv") and ZONA in f]

    print(f"Procesando {len(archivos)} archivos para la zona '{ZONA}' usando {multiprocessing.cpu_count()} núcleos...")

    with multiprocessing.Pool() as pool:
        pool.map(procesar_archivo, archivos)

    print("✅ Todos los archivos procesados.")

    # === Crear GIF con los archivos recién generados ordenados por frecuencia ===
    GIF_FILENAME = os.path.join(CARPETA_OUTPUT, f"{ZONA}_{VAR_TL}_animacion.gif")
    DURACION_ENTRE_FRAMES = 1  # segundos por frame

    # Construir la lista de imágenes generadas con su frecuencia
    imagenes_generadas = []
    for ruta_csv in archivos:
        freq = extraer_frecuencia(os.path.basename(ruta_csv))
        if freq is not None:
            png_filename = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{freq:.1f}Hz_{VAR_TL}.png")
            if os.path.exists(png_filename):
                imagenes_generadas.append((freq, png_filename))

    # Ordenar por frecuencia creciente
    imagenes_generadas.sort(key=lambda x: x[0])
    imagenes_ordenadas = [ruta for _, ruta in imagenes_generadas]

    # Crear el GIF si hay imágenes válidas
    if not imagenes_ordenadas:
        print("⚠️ No se encontraron imágenes generadas para crear el GIF.")
    else:
        with imageio.get_writer(GIF_FILENAME, mode='I', duration=DURACION_ENTRE_FRAMES) as writer:
            for filename in imagenes_ordenadas:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"🎞️ GIF generado correctamente: {GIF_FILENAME}")

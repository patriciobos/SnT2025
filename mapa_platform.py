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

# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "arasj"               # opciones: 'zais', 'gsj', 'arasj'
VAR_TL = "tl_z_8"           # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
FRECUENCIA_OBJETIVO = None  # ejemplo: 100.0 para solo esa frecuencia, o None para procesar todas
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"
UMBRAL_TL_HIGH = 200
UMBRAL_TL_LOW = 50
ALPHA = 0.35
MASCARA_SHP = "Capas/plataforma_continental/plataforma_continentalPolygon.shp"
FILTRO_TL_MIN = 30
PARALELO_NORTE = -26
PARALELO_SUR = -54
PARALELO_INTERPOLACION = -54
MERIDIANO_CORTE = -56

# Coordenadas objetivo con campo opcional "nombre"
coordenadas_objetivo = [
    {"lat": -38.5092, "lon": -56.4850, "nombre": "ZAIS"},
    {"lat": -44.9512, "lon": -63.8894, "nombre": "GSJ"},
    {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},
]
# Ciudades argentinas
ciudades_argentinas = [
    {"nombre": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    #{"nombre": "Montevideo", "lat": -34.9011, "lon": -56.1645},
    {"nombre": "Mar del Plata", "lat": -38.0023, "lon": -57.5575},
    {"nombre": "Bahía Blanca", "lat": -38.7196, "lon": -62.2724},
    {"nombre": "Puerto Madryn", "lat": -42.7692, "lon": -65.0385},
    #{"nombre": "Trelew", "lat": -43.2489, "lon": -65.3051},
    {"nombre": "Comodoro Rivadavia", "lat": -45.8647, "lon": -67.4822},
    {"nombre": "Río Gallegos", "lat": -51.6230, "lon": -69.2168},
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
        
        # Si se especificó una frecuencia y esta no coincide, saltearla
        if FRECUENCIA_OBJETIVO is not None and abs(frecuencia - FRECUENCIA_OBJETIVO) > 0.01:
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

        # Paso uniforme para grid
        step = 3.0

        # === MERIDIANOS (Longitudes) ===
        meridianos = np.arange(np.floor(m.llcrnrlon), np.ceil(m.urcrnrlon) + step, step)
        for i, mer in enumerate(meridianos):
            label = 1 if i % 2 == 0 else 0  # Etiquetar uno de cada dos
            m.drawmeridians([mer], labels=[0, 0, 0, label], linewidth=0.5, color='black')

        # === PARALELOS (Latitudes) ===
        paralelos = np.arange(np.floor(m.llcrnrlat), np.ceil(m.urcrnrlat) + step, step)
        for i, par in enumerate(paralelos):
            label = 1 if i % 2 == 0 else 0  # Etiquetar uno de cada dos
            m.drawparallels([par], labels=[label, 0, 0, 0], linewidth=0.5, color='black')

        for ciudad in ciudades_argentinas:
            cx, cy = m(ciudad["lon"], ciudad["lat"])
            m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
            plt.text(cx, cy, ciudad["nombre"],
                    fontsize=8, ha='right', va='top')


        # Etiqueta "Argentina"
        plt.text(0.15, 0.9, "Argentina", transform=ax.transAxes,
                 fontsize=16, fontweight='bold', color='black',
                 ha='center', va='center', alpha=0.5)


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

        # === FILTRAR REGIÓN SIN DATOS (al sur de Malvinas) ===
        # Excluir puntos: lat < -51 y -63 < lon < -56.5
        mask_excluir = ~((all_valid_points[:, 1] < -51.2) &
                        (all_valid_points[:, 0] > -61.5) &
                        (all_valid_points[:, 0] < -57.5))

        # Aplicar máscara
        all_valid_points = all_valid_points[mask_excluir]

        # === INTERPOLACIÓN ===
        tl_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df[VAR_TL], xi=all_valid_points, method='linear')
        bat_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df['bat'], xi=all_valid_points, method='linear')

        mask_valid = (tl_interp < UMBRAL_TL_HIGH) & (bat_interp > 30)
        final_points = all_valid_points[mask_valid]
        final_tl = tl_interp[mask_valid]

        # === GRAFICAR PUNTOS INTERPOLADOS ===
        x_final, y_final = m(final_points[:, 0], final_points[:, 1])
        sc_interp = m.scatter(x_final, y_final, c=final_tl, cmap='viridis', vmin=UMBRAL_TL_LOW, vmax=UMBRAL_TL_HIGH,
                              marker='s', s=20, edgecolor='none')

        # === GRAFICAR CONTORNO SHP ===
        for geom in gdf_mascara.geometry:
            if geom.geom_type == 'Polygon':
                x_mask, y_mask = m(*geom.exterior.xy)
                ax.plot(x_mask, y_mask, color='gray', linewidth=1.5, label='Argentine Continental Shelf')
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x_mask, y_mask = m(*poly.exterior.xy)
                    ax.plot(x_mask, y_mask, color='gray', linewidth=1.5)

        # === GRAFICAR PUNTO DE CÁLCULO ===
        punto_lon, punto_lat, punto_nombre = obtener_punto_zona(ZONA)
        if punto_lon is not None:
            x_punto, y_punto = m(punto_lon, punto_lat)
            m.plot(x_punto, y_punto, 'r*', markersize=10, label=f'Bouy location: {ZONA.upper()}')
            #plt.text(x_punto + 10000, y_punto + 5000, punto_nombre, fontsize=8, color='red', weight='bold')

        lon_min = np.clip(df['lon'].min(), -70, -50)
        lon_max = np.clip(df['lon'].max(), -70, -50)
        lat_min = np.clip(PARALELO_INTERPOLACION, -55, -25)
        lat_max = np.clip(PARALELO_NORTE, -55, -25)

        # === INLET PLANISFERIO SEGURO ===
        try:
            ax_inlet = fig.add_axes((0.64, 0.68, 0.18, 0.18))  # Arriba a la derecha
            m_inlet = Basemap(projection='cyl',
                  llcrnrlat=-90,
                  urcrnrlat=90,
                  llcrnrlon=-180,
                  urcrnrlon=180,
                  resolution='c',
                  ax=ax_inlet)

            m_inlet.drawcoastlines(linewidth=0.5)
            m_inlet.drawcountries(linewidth=0.5)
            m_inlet.drawmapboundary(fill_color='lightblue')
            m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')

            # Usar el rango limpio ya calculado de interpolación
            rect_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
            rect_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
            m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5, zorder=10)

        except Exception as e:
            print(f"[ERROR] Al crear el inlet planisferio: {e}")


        # === TÍTULO Y GUARDADO ===
        cbar = m.colorbar(sc_interp, location='right', pad="5%")
        cbar.set_label("TL [dB]")
        plt.legend(loc='lower right')
        plt.title(f"Location: {ZONA.upper()} - TL @ {frecuencia} Hz - Z = 8 m")

        os.makedirs(CARPETA_OUTPUT, exist_ok=True)
        output_path = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{frecuencia}Hz_{VAR_TL}.png")
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

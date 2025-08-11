import os
import re
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from shapely.geometry import Point
import alphashape
from scipy.interpolate import griddata
import geopandas as gpd
from geopy.distance import geodesic
import imageio
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.colors import Normalize

# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "arasj"                     # opciones: 'zais', 'gsj', 'arasj'exit
VAR_TL = "tl_z_8"              # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
FRECUENCIA_OBJETIVO = None        # ejemplo: 100.0 para solo esa frecuencia, o None para procesar todas
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"
UMBRAL_TL_HIGH = 200
UMBRAL_TL_LOW = 50
ALPHA = 0.1
MASCARA_SHP = "Capas/plataforma_continental/plataforma_continentalPolygon.shp"
FILTRO_TL_MIN = 1
PARALELO_NORTE = -26
PARALELO_SUR = -54
PARALELO_INTERPOLACION = -54
MERIDIANO_CORTE = -56
PLOT_EXCLUSION_ARCS = True

# === NUEVO: parámetros del sector ===
A_LAT, A_LON = -51.75, -61.53   # Punto A
B_LAT, B_LON = -51.58, -57.37   # Punto B
R_MAX_KM = 2000.0               # rmax

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

def _azimuth(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def _short_arc_center_span(a_deg, b_deg):
    """
    Devuelve (center, halfspan) del arco más corto entre a y b.
    - center en [0,360)
    - halfspan en [0,180]
    """
    # diferencia firmada en (-180,180]
    diff = ((b_deg - a_deg + 540) % 360) - 180
    span = abs(diff)
    center = (a_deg + diff / 2.0) % 360
    halfspan = span / 2.0
    return center, halfspan

def _ang_in_short_arc(theta_deg, a_deg, b_deg):
    """
    True si theta está dentro del arco más corto entre a y b (incluye borde).
    """
    center, halfspan = _short_arc_center_span(a_deg, b_deg)
    # distancia angular mínima a center en [-180,180]
    delta = ((theta_deg - center + 540) % 360) - 180
    return np.abs(delta) <= halfspan

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

        # === PUNTO DE CÁLCULO (lo usamos para el recorte) ===
        punto_lon, punto_lat, punto_nombre = obtener_punto_zona(ZONA)

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
            label = 1 if i % 2 == 0 else 0
            m.drawmeridians([mer], labels=[0, 0, 0, label], linewidth=0.5, color='black')

        # === PARALELOS (Latitudes) ===
        paralelos = np.arange(np.floor(m.llcrnrlat), np.ceil(m.urcrnrlat) + step, step)
        for i, par in enumerate(paralelos):
            label = 1 if i % 2 == 0 else 0
            m.drawparallels([par], labels=[label, 0, 0, 0], linewidth=0.5, color='black')

        for ciudad in ciudades_argentinas:
            cx, cy = m(ciudad["lon"], ciudad["lat"])
            m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
            plt.text(cx, cy, ciudad["nombre"], fontsize=8, ha='right', va='top')

        # Etiqueta "Argentina"
        plt.text(0.15, 0.9, "Argentina", transform=ax.transAxes,
                 fontsize=16, fontweight='bold', color='black',
                 ha='center', va='center', alpha=0.5)

        # === POLÍGONO CÓNCAVO ===
        alpha_shape = alphashape.alphashape(list(zip(df['lon'], df['lat'])), alpha=ALPHA)

        # === MALLA E INTERPOLACIÓN ===
        lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 300)
        lat_grid = np.linspace(PARALELO_INTERPOLACION, PARALELO_NORTE, 300)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.c_[lon_mesh.ravel(), lat_mesh.ravel()]

        # === MÁSCARA SHP (plataforma) + lado derecho del meridiano usando alpha-shape ===
        gdf_mascara = gpd.read_file(MASCARA_SHP).to_crs("EPSG:4326")

        in_alpha = np.array([alpha_shape.contains(Point(lon, lat)) for lon, lat in grid_points])
        points_in_alpha = grid_points[in_alpha & (grid_points[:, 0] > MERIDIANO_CORTE)]

        gdf_grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), crs="EPSG:4326")
        gdf_grid_left = gdf_grid[grid_points[:, 0] <= MERIDIANO_CORTE]
        gdf_grid_left = gpd.sjoin(gdf_grid_left, gdf_mascara, predicate="intersects", how="inner")
        points_in_mask = np.array([[pt.x, pt.y] for pt in gdf_grid_left.geometry])

        # Unimos puntos válidos de ambos lados
        all_valid_points = np.vstack([points_in_alpha, points_in_mask])

        # === EXCLUSIÓN DEL SECTOR ANULAR (ANTES DE INTERPOLAR) ===
        # Conservar: r < rmin  OR  r > rmax  OR  (rmin <= r <= rmax AND FUERA del sector angular corto A-B)
        if (punto_lon is not None) and (punto_lat is not None):
            # Ángulos de A y B desde el centro
            ang_a = _azimuth(punto_lat, punto_lon, A_LAT, A_LON)
            ang_b = _azimuth(punto_lat, punto_lon, B_LAT, B_LON)

            # Radios rmin y rmax
            dist_a = geodesic((punto_lat, punto_lon), (A_LAT, A_LON)).km
            dist_b = geodesic((punto_lat, punto_lon), (B_LAT, B_LON)).km
            r_min_km = (dist_a + dist_b) / 2.0

            lats = all_valid_points[:, 1]
            lons = all_valid_points[:, 0]

            # Ángulo y radio de cada punto
            ang_pts = np.array([_azimuth(punto_lat, punto_lon, lat, lon) for lat, lon in zip(lats, lons)])
            r_pts = np.array([geodesic((punto_lat, punto_lon), (lat, lon)).km for lat, lon in zip(lats, lons)])

            in_annulus = (r_pts >= r_min_km) & (r_pts <= R_MAX_KM)
            in_sector_short = _ang_in_short_arc(ang_pts, ang_a, ang_b)

            # Puntos a ELIMINAR: dentro del anillo Y dentro del sector angular corto A-B
            remove = in_annulus & in_sector_short

            # Puntos a CONSERVAR: todo lo demás
            all_valid_points = all_valid_points[~remove]

        # === INTERPOLACIÓN (ya sin la porción excluida) ===
        tl_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df[VAR_TL], xi=all_valid_points, method='linear')
        bat_interp = griddata(points=np.c_[df['lon'], df['lat']], values=df['bat'], xi=all_valid_points, method='linear')

        mask_valid = (tl_interp < UMBRAL_TL_HIGH) & (bat_interp > 30)
        final_points = all_valid_points[mask_valid]
        final_tl = tl_interp[mask_valid]

        # === GRAFICAR PUNTOS INTERPOLADOS ===
        x_final, y_final = m(final_points[:, 0], final_points[:, 1])
        #sc_interp = m.scatter(x_final, y_final, c=final_tl, cmap='viridis_r', vmin=UMBRAL_TL_LOW, vmax=UMBRAL_TL_HIGH,
        #                      marker='s', s=20, edgecolor='none')

        # Colormap invertido explícito + normalización fija
        cmap_inv = plt.get_cmap('viridis').reversed()
        norm = Normalize(vmin=UMBRAL_TL_LOW, vmax=UMBRAL_TL_HIGH)

        # === GRAFICAR PUNTOS INTERPOLADOS ===
        x_final, y_final = m(final_points[:, 0], final_points[:, 1])
        sc_interp = m.scatter(
            x_final, y_final,
            c=final_tl,
            cmap=cmap_inv,
            norm=norm,
            marker='s', s=20, edgecolor='none'
        )

        # === COLORBAR ===
        cbar = m.colorbar(sc_interp, location='right', pad="5%")
        cbar.set_label("TL [dB]")

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
        if (punto_lon is not None) and (punto_lat is not None):
            x_punto, y_punto = m(punto_lon, punto_lat)
            m.plot(x_punto, y_punto, 'r*', markersize=10, label=f'Buoy location: {ZONA.upper()}')

        # === GRAFICAR ZONA DE EXCLUSIÓN COMPLETA ===
        if PLOT_EXCLUSION_ARCS and (punto_lon is not None) and (punto_lat is not None):
            ang_a = _azimuth(punto_lat, punto_lon, A_LAT, A_LON)
            ang_b = _azimuth(punto_lat, punto_lon, B_LAT, B_LON)

            # rmin según definición
            dist_a = geodesic((punto_lat, punto_lon), (A_LAT, A_LON)).km
            dist_b = geodesic((punto_lat, punto_lon), (B_LAT, B_LON)).km
            r_min_km = (dist_a + dist_b) / 2.0

            # Muestreo de ángulos sobre el arco más corto A-B
            center, halfspan = _short_arc_center_span(ang_a, ang_b)
            angles = (center + np.linspace(-halfspan, halfspan, 180)) % 360

            # --- Arco exterior ---
            arco_max = [geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang) for ang in angles]
            x_max, y_max = m([d.longitude for d in arco_max], [d.latitude for d in arco_max])
            ax.plot(x_max, y_max, linestyle='--', linewidth=1.2, color='red')

            # --- Arco interior ---
            arco_min = [geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang) for ang in angles]
            x_min, y_min = m([d.longitude for d in arco_min], [d.latitude for d in arco_min])
            ax.plot(x_min, y_min, linestyle='--', linewidth=1.2, color='red')

            # --- Radio A ---
            p_a_min = geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang_a)
            p_a_max = geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang_a)
            x_ra, y_ra = m([p_a_min.longitude, p_a_max.longitude], [p_a_min.latitude, p_a_max.latitude])
            ax.plot(x_ra, y_ra, linestyle='--', linewidth=1.2, color='red')

            # --- Radio B ---
            p_b_min = geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang_b)
            p_b_max = geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang_b)
            x_rb, y_rb = m([p_b_min.longitude, p_b_max.longitude], [p_b_min.latitude, p_b_max.latitude])
            ax.plot(x_rb, y_rb, linestyle='--', linewidth=1.2, color='red')

        lon_min = np.clip(df['lon'].min(), -70, -50)
        lon_max = np.clip(df['lon'].max(), -70, -50)
        lat_min = np.clip(PARALELO_INTERPOLACION, -55, -25)
        lat_max = np.clip(PARALELO_NORTE, -55, -25)

        # === INLET PLANISFERIO SEGURO ===
        try:
            ax_inlet = fig.add_axes((0.64, 0.68, 0.18, 0.18))  # Arriba a la derecha
            m_inlet = Basemap(projection='cyl',
                  llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,
                  resolution='c', ax=ax_inlet)

            m_inlet.drawcoastlines(linewidth=0.5)
            m_inlet.drawcountries(linewidth=0.5)
            m_inlet.drawmapboundary(fill_color='lightblue')
            m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')

            rect_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
            rect_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
            m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5, zorder=10)

        except Exception as e:
            print(f"[ERROR] Al crear el inlet planisferio: {e}")

        # === TÍTULO Y GUARDADO ===
        cbar = m.colorbar(sc_interp, location='right', pad="5%")
        cbar.set_label("TL [dB]")
        plt.legend(loc='lower right')
        plt.title(f"Location: {ZONA.upper()} - TL @ {frecuencia} Hz - Z = 8 m.")

        os.makedirs(CARPETA_OUTPUT, exist_ok=True)
        output_path = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{frecuencia}Hz_{VAR_TL}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Procesado {nombre_archivo} → {output_path}")

    except Exception as e:
        print(f"[ERROR] Al procesar {ruta_archivo}: {e}")

def crear_gif(carpeta_output, zona, var_tl, duracion=1.0):
    imagenes = [
        os.path.join(carpeta_output, f)
        for f in os.listdir(carpeta_output)
        if f.endswith(".png") and f.startswith(f"{zona}_f") and var_tl in f
    ]
    if not imagenes:
        print("⚠️ No se encontraron imágenes para generar el GIF.")
        return

    imagenes.sort(key=lambda f: extraer_frecuencia(os.path.basename(f)) or float('inf'))
    gif_path = os.path.join(carpeta_output, f"{zona}_{var_tl}.gif")

    print(f"🎞️ Generando GIF con {len(imagenes)} imágenes...")
    frames = [imageio.v2.imread(img) for img in imagenes]
    imageio.mimsave(gif_path, frames, duration=duracion)
    print(f"✅ GIF generado: {gif_path}")

if __name__ == "__main__":
    archivos = [os.path.join(CARPETA_INPUT, f)
                for f in os.listdir(CARPETA_INPUT)
                if f.endswith(".csv") and ZONA in f]

    print(f"Procesando {len(archivos)} archivos para la zona '{ZONA}' usando {multiprocessing.cpu_count()} núcleos...")

    with multiprocessing.Pool() as pool:
        pool.map(procesar_archivo, archivos)

    print("✅ Todos los archivos procesados.")
    crear_gif(CARPETA_OUTPUT, ZONA, VAR_TL, duracion=1.0)

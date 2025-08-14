# === Estándar de Python ===
import os
import re
import multiprocessing
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.interpolate import griddata
import geopandas as gpd
from geopy.distance import geodesic
import imageio
from matplotlib.colors import Normalize

# ==== CONFIGURACIÓN DEL USUARIO ====
ZONA = "gsj"                     # opciones: 'zais', 'gsj', 'arasj'
VAR_TL = "tl_z_8"                 # opciones: 'tl_z_8', 'tl_z_half', 'tl_max_z'
FRECUENCIA_OBJETIVO = None        # ejemplo: 100.0 para solo esa frecuencia, o None para procesar todas
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"
UMBRAL_TL_HIGH = 200
UMBRAL_TL_LOW = 50
MASCARA_SHP = "Capas/plataforma_continental/plataforma_continentalPolygon.shp"
FILTRO_TL_MIN = 1
MERIDIANO_CORTE = -56.0           # a la izquierda: recorte por plataforma continental
PLOT_EXCLUSION_ARCS = True


# === EXCLUSIÓN POR LISTA DE PUNTOS (excluir.csv en CARPETA_OUTPUT) ===
EXCLUSIONES_CSV = "excluir.csv"     # archivo dentro de CARPETA_OUTPUT (p.ej., 'mapas/excluir.csv')
EXCLUSION_RADIUS_KM_DEFAULT = 10.0  # radio por defecto alrededor de cada (lat, lon)
EXCLUSION_MATCH_TL_TOL_DB = None    # si querés exigir coincidencia de TL ±tolerancia (ej.: 3.0); None = ignorar TL del CSV


# Alias SOLO para mostrar en título y leyenda
ALIAS_ZONA = {
    "zais": "MDQ",   # si ZONA == 'zais' → mostrar 'MDQ'
    # podés agregar más, ej: "gsj": "GSJ"
}

def nombre_para_mostrar(zona: str) -> str:
    """Devuelve el nombre a mostrar (usa alias si existe; si no, ZONA.upper())."""
    return ALIAS_ZONA.get(zona.lower(), zona.upper())


# === Parámetros del sector A-B (opcional, para exclusión angular) ===
A_LAT, A_LON = -51.75, -61.53   # Punto A
B_LAT, B_LON = -51.58, -57.37   # Punto B
R_MAX_KM = 2000.0               # rmax para la exclusión angular

# === LÍMITES FIJOS DE MAPA (iguales para todas las figuras) ===
MAP_LL_LAT, MAP_UR_LAT = -55.0, -30.0   # Sur, Norte
MAP_LL_LON, MAP_UR_LON = -70.0, -35.0   # Oeste, Este

# === MÁSCARA ÚNICA: círculo centrado en GSJ de 2200 km ===
GSJ_LAT, GSJ_LON = -44.9512, -63.8894
CIRCLE_R_KM = 2200.0
EARTH_R_KM = 6371.0088

# Coordenadas objetivo con campo opcional "nombre"
coordenadas_objetivo = [
    {"lat": -38.5092, "lon": -56.4850, "nombre": "ZAIS"},
    {"lat": -44.9512, "lon": -63.8894, "nombre": "GSJ"},
    {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},
]

# Ciudades argentinas
ciudades_argentinas = [
    {"nombre": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    {"nombre": "Mar del Plata", "lat": -38.0023, "lon": -57.5575},
    {"nombre": "Bahía Blanca", "lat": -38.7196, "lon": -62.2724},
    {"nombre": "Puerto Madryn", "lat": -42.7692, "lon": -65.0385},
    {"nombre": "Comodoro Rivadavia", "lat": -45.8647, "lon": -67.4822},
    {"nombre": "Río Gallegos", "lat": -51.6230, "lon": -69.2168},
]

# ==== UTILIDADES GLOBALES ====

def _dist_km_to(lat, lon, lat0, lon0):
    """Distancia Haversine vectorizada (km)."""
    lat = np.radians(lat); lon = np.radians(lon)
    lat0 = np.radians(lat0); lon0 = np.radians(lon0)
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat/2.0)**2 + np.cos(lat0)*np.cos(lat)*np.sin(dlon/2.0)**2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(a))

def extraer_frecuencia(nombre_archivo):
    """Busca 'f<numero>(.decimal)?Hz' en el nombre y devuelve float o None."""
    match = re.search(r"f(\d+(?:\.\d+)?)\s*Hz", nombre_archivo)
    return float(match.group(1)) if match else None

def obtener_punto_zona(zona):
    for entry in coordenadas_objetivo:
        if entry["nombre"].lower() == zona.lower():
            return entry["lon"], entry["lat"], entry["nombre"]
    return None, None, None

def _azimuth(lat1, lon1, lat2, lon2):
    """Acimut (grados 0-360) de (lat1,lon1) hacia (lat2,lon2). Soporta arrays en lat2/lon2."""
    dlon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def _short_arc_center_span(a_deg, b_deg):
    """Devuelve (center, halfspan) del arco más corto entre a y b."""
    diff = ((b_deg - a_deg + 540) % 360) - 180
    span = abs(diff)
    center = (a_deg + diff / 2.0) % 360
    halfspan = span / 2.0
    return center, halfspan

def _ang_in_short_arc(theta_deg, a_deg, b_deg):
    """True si theta (array) cae en el arco más corto entre a y b (incluye borde)."""
    center, halfspan = _short_arc_center_span(a_deg, b_deg)
    delta = ((theta_deg - center + 540) % 360) - 180
    return np.abs(delta) <= halfspan

# Cargar y preparar la geometría de la plataforma una sola vez
try:
    _gdf_plat = gpd.read_file(MASCARA_SHP).to_crs("EPSG:4326")
    PLATAFORMA_GEOM = unary_union(_gdf_plat.geometry)
    PLATAFORMA_PREP = prep(PLATAFORMA_GEOM)  # (no todas las predicados están preparados, usar con cuidado)
except Exception as e:
    print(f"[ERROR] No se pudo leer la máscara de plataforma: {e}")
    PLATAFORMA_GEOM = None
    PLATAFORMA_PREP = None


# ==== EXCLUSIONES DESDE CSV (se carga una sola vez) ====
EXCLUSIONES = None  # np.ndarray de shape (N, 2) con [lat, lon]
EXCLUSIONES_R = None  # np.ndarray opcional con radios por fila
EXCLUSIONES_TL = None  # np.ndarray opcional con TL por fila (si lo querés usar)

try:
    path_exclusiones = os.path.join(CARPETA_OUTPUT, EXCLUSIONES_CSV)
    if os.path.isfile(path_exclusiones):
        _df_exc = pd.read_csv(path_exclusiones)
        cols_ok = {'lat', 'lon'}
        if not cols_ok.issubset({c.lower() for c in _df_exc.columns}):
            raise ValueError("excluir.csv debe tener columnas 'lat' y 'lon' (y opcionalmente 'tl' y 'r_km').")

        # normalizo nombres por si vienen con mayúsculas
        _df_exc = _df_exc.rename(columns={c: c.lower() for c in _df_exc.columns})

        EXCLUSIONES = _df_exc[['lat', 'lon']].to_numpy(dtype=float)

        if 'r_km' in _df_exc.columns:
            EXCLUSIONES_R = _df_exc['r_km'].astype(float).to_numpy()
        if 'tl' in _df_exc.columns:
            EXCLUSIONES_TL = _df_exc['tl'].astype(float).to_numpy()

        print(f"[INFO] Cargadas {len(EXCLUSIONES)} ubicaciones de exclusión desde {path_exclusiones}")
    else:
        print(f"[INFO] No se encontró {path_exclusiones}; no se aplicará exclusión por lista.")
except Exception as e:
    print(f"[WARN] No se pudo leer exclusiones desde CSV: {e}")
    EXCLUSIONES = None
    EXCLUSIONES_R = None
    EXCLUSIONES_TL = None


# ==== PROCESAMIENTO ====

def procesar_archivo(ruta_archivo):
    try:
        nombre_archivo = os.path.basename(ruta_archivo)
        if ZONA not in nombre_archivo:
            return
    
        display_zona = nombre_para_mostrar(ZONA)

        frecuencia = extraer_frecuencia(nombre_archivo)
        if frecuencia is None:
            print(f"[WARN] No se pudo extraer la frecuencia de {nombre_archivo}")
            return

        # Filtro por frecuencia objetivo (si corresponde)
        if FRECUENCIA_OBJETIVO is not None and abs(frecuencia - FRECUENCIA_OBJETIVO) > 0.01:
            return

        # === LECTURA Y FILTROS BÁSICOS ===
        df = pd.read_csv(ruta_archivo)
        df = df[df[VAR_TL] > FILTRO_TL_MIN]

        columnas_necesarias = {'lat', 'lon', VAR_TL, 'bat'}
        if not columnas_necesarias.issubset(df.columns):
            print(f"[ERROR] Columnas faltantes en {nombre_archivo}")
            return

        # === EXCLUSIÓN POR LISTA DE PUNTOS (excluir.csv) ===
        if EXCLUSIONES is not None and len(EXCLUSIONES) > 0:
            lat_arr = df['lat'].to_numpy(dtype=float)
            lon_arr = df['lon'].to_numpy(dtype=float)
            tl_arr = df[VAR_TL].to_numpy(dtype=float)

            # máscara de filas a excluir (se va acumulando)
            mask_excluir = np.zeros(len(df), dtype=bool)

            for i, (clat, clon) in enumerate(EXCLUSIONES):
                r_km = (
                    EXCLUSIONES_R[i]
                    if EXCLUSIONES_R is not None and i < len(EXCLUSIONES_R) and np.isfinite(EXCLUSIONES_R[i])
                    else EXCLUSION_RADIUS_KM_DEFAULT
                )

                # distancia Haversine vectorizada a este centro
                d_km = _dist_km_to(lat_arr, lon_arr, clat, clon)

                if EXCLUSION_MATCH_TL_TOL_DB is not None and EXCLUSIONES_TL is not None and i < len(EXCLUSIONES_TL):
                    tl0 = EXCLUSIONES_TL[i]
                    mask_i = (d_km <= r_km) & (np.abs(tl_arr - tl0) <= EXCLUSION_MATCH_TL_TOL_DB)
                else:
                    mask_i = (d_km <= r_km)

                mask_excluir |= mask_i

            if np.any(mask_excluir):
                excluidas = int(mask_excluir.sum())
                total = len(df)
                df = df[~mask_excluir].copy()
                print(f"[INFO] Excluidas {excluidas}/{total} filas por proximidad a excluir.csv")

        # === PUNTO DE CÁLCULO (para sector de exclusión) ===
        punto_lon, punto_lat, punto_nombre = obtener_punto_zona(ZONA)

        # === FIGURA Y MAPA CON LÍMITES FIJOS ===
        fig, ax = plt.subplots(figsize=(10, 8))
        m = Basemap(
            projection='merc',
            llcrnrlat=MAP_LL_LAT, urcrnrlat=MAP_UR_LAT,
            llcrnrlon=MAP_LL_LON, urcrnrlon=MAP_UR_LON,
            resolution='i', ax=ax
        )

        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgray', lake_color='lightblue')

        # === MERIDIANOS / PARALELOS ===
        step = 3.0
        meridianos = np.arange(MAP_LL_LON, MAP_UR_LON + step, step)
        for i, mer in enumerate(meridianos):
            label = 1 if i % 2 == 0 else 0
            m.drawmeridians([mer], labels=[0, 0, 0, label], linewidth=0.5, color='black')

        paralelos = np.arange(MAP_LL_LAT, MAP_UR_LAT + step, step)
        for i, par in enumerate(paralelos):
            label = 1 if i % 2 == 0 else 0
            m.drawparallels([par], labels=[label, 0, 0, 0], linewidth=0.5, color='black')

        # === MARCADORES DE CIUDADES ===
        for ciudad in ciudades_argentinas:
            cx, cy = m(ciudad["lon"], ciudad["lat"])
            m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
            plt.text(cx, cy, ciudad["nombre"], fontsize=8, ha='right', va='top')

        # Etiqueta "Argentina"
        plt.text(0.15, 0.9, "Argentina", transform=ax.transAxes,
                 fontsize=16, fontweight='bold', color='black',
                 ha='center', va='center', alpha=0.5)

        # === MALLA FIJA (igual para todos los mapas) ===
        lon_grid = np.linspace(MAP_LL_LON, MAP_UR_LON, 300)
        lat_grid = np.linspace(MAP_LL_LAT, MAP_UR_LAT, 300)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # === INTERPOLACIÓN SOBRE TODA LA MALLA ===
        grid_TL = griddata(points=np.c_[df['lon'], df['lat']], values=df[VAR_TL],
                           xi=(lon_mesh, lat_mesh), method='linear')
        grid_bat = griddata(points=np.c_[df['lon'], df['lat']], values=df['bat'],
                            xi=(lon_mesh, lat_mesh), method='linear')

        # === MÁSCARA 1: CÍRCULO ÚNICO (GSJ, 2200 km) ===
        dist_gsj = _dist_km_to(lat_mesh, lon_mesh, GSJ_LAT, GSJ_LON)
        mask_circle = dist_gsj <= CIRCLE_R_KM

        # === MÁSCARA 2: PLATAFORMA A LA IZQUIERDA DE MERIDIANO_CORTE ===
        mask_left = lon_mesh <= MERIDIANO_CORTE
        if PLATAFORMA_GEOM is not None:
            flat_lon = lon_mesh.ravel()
            flat_lat = lat_mesh.ravel()
            # covers() incluye los bordes (si no está disponible, usar contains() | touches())
            mask_plat_flat = np.array([PLATAFORMA_GEOM.covers(Point(x, y)) for x, y in zip(flat_lon, flat_lat)])
            mask_plataforma = mask_plat_flat.reshape(lon_mesh.shape)
        else:
            mask_plataforma = np.ones_like(mask_left, dtype=bool)

        # A la izquierda del meridiano: exigir plataforma; a la derecha: no aplicar plataforma
        mask_geo = np.where(mask_left, mask_plataforma, True)

        # === (OPCIONAL) EXCLUSIÓN DEL SECTOR A–B ===
        remove = np.zeros_like(mask_circle, dtype=bool)
        if (punto_lon is not None) and (punto_lat is not None):
            ang_a = _azimuth(punto_lat, punto_lon, A_LAT, A_LON)
            ang_b = _azimuth(punto_lat, punto_lon, B_LAT, B_LON)

            dist_a = geodesic((punto_lat, punto_lon), (A_LAT, A_LON)).km
            dist_b = geodesic((punto_lat, punto_lon), (B_LAT, B_LON)).km
            r_min_km = (dist_a + dist_b) / 2.0

            ang_pts = _azimuth(punto_lat, punto_lon, lat_mesh, lon_mesh)  # vectorizado
            r_pts = _dist_km_to(lat_mesh, lon_mesh, punto_lat, punto_lon)

            in_annulus = (r_pts >= r_min_km) & (r_pts <= R_MAX_KM)
            in_sector_short = _ang_in_short_arc(ang_pts, ang_a, ang_b)
            remove = in_annulus & in_sector_short

        # === MÁSCARA FINAL ===
        mask_valid = (
            mask_circle
            & mask_geo
            & ~remove
            & (grid_TL < UMBRAL_TL_HIGH)
            & (grid_bat > 30)
            & ~np.isnan(grid_TL)
        )

        # === PLOTEO (scatter en cuadrícula) con colormap invertido y escala fija ===
        cmap_inv = plt.get_cmap('viridis').reversed()
        norm = Normalize(vmin=UMBRAL_TL_LOW, vmax=UMBRAL_TL_HIGH)

        x_plot, y_plot = m(lon_mesh[mask_valid], lat_mesh[mask_valid])
        sc_interp = m.scatter(
            x_plot, y_plot,
            c=grid_TL[mask_valid],
            cmap=cmap_inv,
            norm=norm,
            marker='s', s=20, edgecolor='none'
        )

        # === CONTORNO SHP (con leyenda simple) ===
        try:
            # 1) Crear la entrada de leyenda una sola vez (dummy handle):
            ax.plot([], [], color='gray', linewidth=1.5, label='Argentine Continental Shelf')

            # 2) Dibujar el/los contornos sin label:
            gdf_mascara = gpd.read_file(MASCARA_SHP).to_crs("EPSG:4326")
            for geom in gdf_mascara.geometry:
                if geom.geom_type == 'Polygon':
                    x, y = m(*geom.exterior.xy)
                    ax.plot(x, y, color='gray', linewidth=1.5)
                    # (opcional) agujeros internos
                    for ring in geom.interiors:
                        xi, yi = m(*ring.xy)
                        ax.plot(xi, yi, color='gray', linewidth=1.0)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = m(*poly.exterior.xy)
                        ax.plot(x, y, color='gray', linewidth=1.5)
                        for ring in poly.interiors:
                            xi, yi = m(*ring.xy)
                            ax.plot(xi, yi, color='gray', linewidth=1.0)
        except Exception as e:
            print(f"[WARN] No se pudo dibujar el SHP de plataforma: {e}")


        # === PUNTO DE CÁLCULO ===
        if (punto_lon is not None) and (punto_lat is not None):
            x_punto, y_punto = m(punto_lon, punto_lat)
            m.plot(x_punto, y_punto, 'r*', markersize=10, label=f'Buoy location: {display_zona}')


        # === ARCOS/SECTOR DE EXCLUSIÓN (trazado) ===
        if PLOT_EXCLUSION_ARCS and (punto_lon is not None) and (punto_lat is not None):
            ang_a = _azimuth(punto_lat, punto_lon, A_LAT, A_LON)
            ang_b = _azimuth(punto_lat, punto_lon, B_LAT, B_LON)

            dist_a = geodesic((punto_lat, punto_lon), (A_LAT, A_LON)).km
            dist_b = geodesic((punto_lat, punto_lon), (B_LAT, B_LON)).km
            r_min_km = (dist_a + dist_b) / 2.0

            center, halfspan = _short_arc_center_span(ang_a, ang_b)
            angles = (center + np.linspace(-halfspan, halfspan, 180)) % 360

            # Arco exterior (R_MAX_KM)
            arco_max = [geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang) for ang in angles]
            x_max, y_max = m([d.longitude for d in arco_max], [d.latitude for d in arco_max])
            ax.plot(x_max, y_max, linestyle='--', linewidth=1.2, color='red')

            # Arco interior (r_min_km)
            arco_min = [geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang) for ang in angles]
            x_min, y_min = m([d.longitude for d in arco_min], [d.latitude for d in arco_min])
            ax.plot(x_min, y_min, linestyle='--', linewidth=1.2, color='red')

            # Radios
            p_a_min = geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang_a)
            p_a_max = geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang_a)
            x_ra, y_ra = m([p_a_min.longitude, p_a_max.longitude], [p_a_min.latitude, p_a_max.latitude])
            ax.plot(x_ra, y_ra, linestyle='--', linewidth=1.2, color='red')

            p_b_min = geodesic(kilometers=r_min_km).destination((punto_lat, punto_lon), ang_b)
            p_b_max = geodesic(kilometers=R_MAX_KM).destination((punto_lat, punto_lon), ang_b)
            x_rb, y_rb = m([p_b_min.longitude, p_b_max.longitude], [p_b_min.latitude, p_b_max.latitude])
            ax.plot(x_rb, y_rb, linestyle='--', linewidth=1.2, color='red')

        # === INLET PLANISFERIO (rectángulo = límites fijos) ===
        try:
            ax_inlet = fig.add_axes((0.6, 0.72, 0.18, 0.18))
            m_inlet = Basemap(projection='cyl',
                              llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,
                              resolution='c', ax=ax_inlet)
            m_inlet.drawcoastlines(linewidth=0.5)
            m_inlet.drawcountries(linewidth=0.5)
            m_inlet.drawmapboundary(fill_color='lightblue')
            m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')

            rect_lons = [MAP_LL_LON, MAP_UR_LON, MAP_UR_LON, MAP_LL_LON, MAP_LL_LON]
            rect_lats = [MAP_LL_LAT, MAP_LL_LAT, MAP_UR_LAT, MAP_UR_LAT, MAP_LL_LAT]
            m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5, zorder=10)
        except Exception as e:
            print(f"[ERROR] Al crear el inlet planisferio: {e}")

        # === COLORBAR, LEYENDA, TÍTULO, GUARDADO ===
        cbar = m.colorbar(sc_interp, location='right', pad="5%")
        cbar.set_label("TL [dB]")
        plt.legend(loc='lower right')
        plt.title(f"Location: {display_zona} - TL at {frecuencia} Hz - Z = 8 m.", fontsize=18)


        os.makedirs(CARPETA_OUTPUT, exist_ok=True)
        output_path = os.path.join(CARPETA_OUTPUT, f"{ZONA}_f{frecuencia}Hz_{VAR_TL}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Procesado {nombre_archivo} → {output_path}")

    except Exception as e:
        print(f"[ERROR] Al procesar {ruta_archivo}: {e}")

# ==== EXPORTS: GIF y MP4 ====

def crear_gif(carpeta_output, zona, var_tl, duracion=1.0):
    """Genera un GIF (loop infinito) ordenando por frecuencia extraída del nombre."""
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
    imageio.mimsave(gif_path, frames, duration=duracion)  # loop infinito por defecto en la mayoría de visores
    print(f"✅ GIF generado: {gif_path}")

def crear_mp4(carpeta_output, zona, var_tl, fps=24):
    """Genera un MP4 (una pasada) con ffmpeg. Loop/autoplay se configuran en Impress."""
    imagenes = [
        os.path.join(carpeta_output, f)
        for f in os.listdir(carpeta_output)
        if f.endswith(".png") and f.startswith(f"{zona}_f") and var_tl in f
    ]
    if not imagenes:
        print("⚠️ No se encontraron imágenes para generar el MP4.")
        return

    imagenes.sort(key=lambda f: extraer_frecuencia(os.path.basename(f)) or float('inf'))
    mp4_path = os.path.join(carpeta_output, f"{zona}_{var_tl}.mp4")

    with tempfile.TemporaryDirectory() as tdir:
        list_path = Path(tdir) / "list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for img in imagenes:
                f.write(f"file '{os.path.abspath(img)}'\n")
                f.write(f"duration {1.0/fps:.10f}\n")
            # ffmpeg ignora la duración del último → repetir último frame
            f.write(f"file '{os.path.abspath(imagenes[-1])}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}",
            mp4_path,
        ]
        print(f"🎬 Generando MP4 ({len(imagenes)} imágenes, {fps} fps)...")
        subprocess.run(cmd, check=True)
        print(f"✅ MP4 generado: {mp4_path}\n   (Activá 'Repetir hasta detener' e 'Iniciar automáticamente' en Impress si querés loop/autoplay)")

# ==== MAIN ====

if __name__ == "__main__":
    # Crear carpeta de salida si no existe
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)

    archivos = [os.path.join(CARPETA_INPUT, f)
                for f in os.listdir(CARPETA_INPUT)
                if f.endswith(".csv") and ZONA in f]

    print(f"Procesando {len(archivos)} archivos para la zona '{ZONA}' usando {multiprocessing.cpu_count()} núcleos...")

    with multiprocessing.Pool() as pool:
        pool.map(procesar_archivo, archivos)

    print("✅ Todos los archivos procesados.")
    # GIF (loop infinito) y MP4 (una pasada)
    crear_gif(CARPETA_OUTPUT, ZONA, VAR_TL, duracion=1.0)
    crear_mp4(CARPETA_OUTPUT, ZONA, VAR_TL, fps=24)

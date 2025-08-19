# === Estándar de Python ===
import os
import re
import glob
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point

import subprocess
import tempfile
from pathlib import Path
from PIL import Image  # pip install pillow


def procesar_archivo(filename):
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from mpl_toolkits.basemap import Basemap
    from scipy.interpolate import griddata
    import alphashape
    import geopandas as gpd
    from shapely.ops import unary_union
    from shapely.geometry import Point

    # --- nombres y metadatos ---
    basename = os.path.basename(filename)
    match = re.search(r"f([\d.]+)\s*Hz", basename)
    frecuencia_str = f"{match.group(1)} Hz" if match else "Frecuencia desconocida"
    frecuencia_limpia = match.group(1).replace('.', '_') if match else "desconocida"
    nombre_figura = f"mapa_tl_z_8_f{frecuencia_limpia}"

    print(f"Procesando: {basename}")

    try:
        # --- lectura y limpieza de datos ---
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip().str.lower()

        columnas_TL = ['tl_z_8']
        for col in ['lat', 'lon', 'bat'] + columnas_TL:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
        df['lat'] = df['lat'].astype(float)
        df['lon'] = df['lon'].astype(float)
        df['bat'] = -df['bat'].astype(float)

        # --- filtro de TL ---
        TLmax = 200
        col_TL = columnas_TL[0]
        #df_filtrado = df.copy()  # no descartamos >=200; luego clip para saturar
        df_filtrado = df[df[col_TL] < TLmax].copy()
        lat_f = df_filtrado['lat'].values
        lon_f = df_filtrado['lon'].values
        TL = df_filtrado[col_TL].clip(upper=TLmax).values

        # --- malla e interpolación ---
        grid_lat = np.linspace(np.min(lat_f), np.max(lat_f), 300)
        grid_lon = np.linspace(np.min(lon_f), np.max(lon_f), 300)
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
        points = np.column_stack((lon_f, lat_f))
        grid_TL = griddata(points, TL, (grid_lon2d, grid_lat2d), method='linear')

        # ---------------------------------------------------------------------
        # NUEVO (previo): máscara = intersección (alpha shape ∩ plataforma continental)
        # ---------------------------------------------------------------------
        alpha = 3.0
        concave_hull = alphashape.alphashape(points, alpha)
        shp_path = "Capas/plataforma_continental/plataforma_continentalPolygon.shp"
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        plataforma_union = unary_union(gdf.geometry)
        area_intersectada = concave_hull.intersection(plataforma_union)

        flat_lon = grid_lon2d.ravel()
        flat_lat = grid_lat2d.ravel()

        if area_intersectada.is_empty:
            grid_TL_masked = np.full_like(grid_TL, np.nan)
        else:
            mask = np.array([area_intersectada.covers(Point(x, y)) for x, y in zip(flat_lon, flat_lat)])
            grid_TL_masked = np.full_like(grid_TL, np.nan)
            grid_TL_masked.ravel()[mask] = grid_TL.ravel()[mask]

        # === FIGURA 1: mapa interpolado ===
        fig = plt.figure(figsize=(10, 9))
        ax_main = fig.add_axes((0.1, 0.1, 0.85, 0.85))
        m = Basemap(projection='merc',
                    llcrnrlat=-55, urcrnrlat=-35,
                    llcrnrlon=-70, urcrnrlon=-45,
                    resolution='i', ax=ax_main)

        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawparallels(range(-55, -34, 5), labels=[1, 0, 0, 0])
        m.drawmeridians(range(-70, -44, 5), labels=[0, 0, 0, 1])

        shapefile_base = "Capas/plataforma_continental/plataforma_continentalPolygon"
        m.readshapefile(shapefile_base, 'plataforma', drawbounds=True, color='gray', linewidth=1.0)

        x, y = m(grid_lon2d, grid_lat2d)
        norm = Normalize(vmin=50, vmax=200)
        cmap = colormaps['viridis_r']
        im = m.pcolormesh(x, y, grid_TL_masked, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(im, ax=ax_main, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label(f"{col_TL} (dB)")

        ciudades_argentinas = [
            {"nombre": "Mar del Plata", "lat": -38.0023, "lon": -57.5575},
            {"nombre": "Bahía Blanca", "lat": -38.7196, "lon": -62.2724},
            {"nombre": "Puerto Madryn", "lat": -42.7692, "lon": -65.0385},
            {"nombre": "Trelew", "lat": -43.2489, "lon": -65.3051},
            {"nombre": "Comodoro Rivadavia", "lat": -45.8647, "lon": -67.4822},
            {"nombre": "Río Gallegos", "lat": -51.6230, "lon": -69.2168},
        ]
        for ciudad in ciudades_argentinas:
            cx, cy = m(ciudad["lon"], ciudad["lat"])
            m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
            plt.text(cx + 5000, cy + 5000, ciudad["nombre"], fontsize=8, ha='center', va='bottom')

        plt.text(0.15, 0.9, "Argentina", transform=ax_main.transAxes,
                 fontsize=16, fontweight='bold', color='black',
                 ha='center', va='center', alpha=0.5)

        coordenadas_objetivo = [
            {"lat": -38.5092, "lon": -56.4850, "nombre": "MDQ"},
            {"lat": -44.9512, "lon": -63.8894, "nombre": "CRD"},
            {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},
        ]
        for punto in coordenadas_objetivo:
            px, py = m(punto["lon"], punto["lat"])
            m.plot(px, py, marker='*', color='red', markersize=8, zorder=6)
            plt.text(px + 5000, py + 5000, punto["nombre"], fontsize=9, ha='left', va='bottom', color='red')

        ax_inlet = fig.add_axes([0.57, 0.065, 0.22, 0.22])
        m_inlet = Basemap(projection='cyl', resolution='c', ax=ax_inlet)
        m_inlet.drawcoastlines(linewidth=0.5)
        m_inlet.drawcountries(linewidth=0.5)
        m_inlet.drawmapboundary(fill_color='lightblue')
        m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')
        rect_lons = [-70, -45, -45, -70, -70]
        rect_lats = [-55, -55, -35, -35, -55]
        m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5)

        ax_main.set_title(f"Transmission Loss from H10N f = {frecuencia_str}, Z = 8 m.", fontsize=18)

        os.makedirs("figuras", exist_ok=True)
        fig.savefig(f"figuras/{nombre_figura}_basemap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        return f"{basename} OK (mapas guardados)"

    except Exception as e:
        return f"{basename} ERROR: {e}"


# === Resto del script (sin cambios) ===

def extraer_frecuencia_desde_png(nombre: str) -> float | None:
    base = os.path.basename(nombre)
    m = re.search(r"_f(\d+(?:_\d+)?)", base)
    if not m:
        return None
    freq_txt = m.group(1).replace("_", ".")
    try:
        return float(freq_txt)
    except ValueError:
        return None


def recolectar_frames_figuras(carpeta="figuras") -> list[str]:
    patrones = [
        os.path.join(carpeta, "mapa_tl_z_8_f*_basemap.png"),
        os.path.join(carpeta, "mapa_*_f*_basemap.png"),
    ]
    archivos = []
    for patron in patrones:
        archivos.extend(glob.glob(patron))

    pares = []
    for f in archivos:
        fr = extraer_frecuencia_desde_png(f)
        if fr is not None:
            pares.append((fr, f))

    pares.sort(key=lambda t: t[0])
    return [f for _, f in pares]


def crear_gif(frames: list[str], salida_gif: str, fps: int = 24, optimizar: bool = True):
    if not frames:
        raise ValueError("No hay frames para el GIF.")
    imgs = [Image.open(f).convert("RGB") for f in frames]
    if optimizar:
        imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in imgs]
    dur_ms = int(1000 / fps)
    imgs[0].save(
        salida_gif,
        save_all=True,
        append_images=imgs[1:],
        duration=dur_ms,
        loop=0,
        disposal=2,
        optimize=True
    )


def crear_mp4_con_ffmpeg(frames: list[str], salida_mp4: str, fps: int = 24):
    if not frames:
        raise ValueError("No hay frames para el MP4.")
    with tempfile.TemporaryDirectory() as tdir:
        list_path = Path(tdir) / "list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for path in frames:
                f.write(f"file '{os.path.abspath(path)}'\n")
                f.write(f"duration {1.0/fps:.10f}\n")
            f.write(f"file '{os.path.abspath(frames[-1])}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}",
            salida_mp4,
        ]
        subprocess.run(cmd, check=True)


def crear_animaciones(carpeta_fig="figuras", fps=24):
    frames = recolectar_frames_figuras(carpeta_fig)
    if not frames:
        print("No se encontraron frames PNG en", carpeta_fig)
        return None
    os.makedirs(carpeta_fig, exist_ok=True)
    gif_path = os.path.join(carpeta_fig, "animacion_TL.gif")
    mp4_path = os.path.join(carpeta_fig, "animacion_TL.mp4")
    print(f"Creando GIF ({len(frames)} frames) → {gif_path}")
    crear_gif(frames, gif_path, fps=fps, optimizar=True)
    print(f"Creando MP4 (una pasada) → {mp4_path}")
    crear_mp4_con_ffmpeg(frames, mp4_path, fps=fps)
    print("Listo. En Impress: activar 'Repetir hasta detener' e 'Iniciar automáticamente' si querés loop/autoplay.")
    return {"gif": gif_path, "mp4": mp4_path, "frames": len(frames)}

def crear_mapa_datapoints_desde_csv(csv_path: str, salida: str = "figuras/mapa_datapoints_unico.png") -> str:
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    for col in ['lat', 'lon']:
        df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
    lats = df['lat'].astype(float).values
    lons = df['lon'].astype(float).values

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.85))
    m = Basemap(projection='merc', llcrnrlat=-55, urcrnrlat=-35, llcrnrlon=-70, urcrnrlon=-45, resolution='i', ax=ax)
    m.drawcoastlines(); m.drawcountries(); m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawparallels(range(-55, -34, 5), labels=[1,0,0,0]); m.drawmeridians(range(-70, -44, 5), labels=[0,0,0,1])

    try:
        m.readshapefile("Capas/plataforma_continental/plataforma_continentalPolygon", 'plataforma', drawbounds=True, color='gray', linewidth=1.0)
    except Exception:
        pass

    x, y = m(lons, lats)
    m.scatter(x, y, s=2, c='gray', edgecolors='black', linewidths=0.3, marker='o', label='Data points', zorder=7)

    ciudades_argentinas = [
        {"nombre": "Mar del Plata", "lat": -38.0023, "lon": -57.5575},
        {"nombre": "Bahía Blanca", "lat": -38.7196, "lon": -62.2724},
        {"nombre": "Puerto Madryn", "lat": -42.7692, "lon": -65.0385},
        {"nombre": "Trelew", "lat": -43.2489, "lon": -65.3051},
        {"nombre": "Comodoro Rivadavia", "lat": -45.8647, "lon": -67.4822},
        {"nombre": "Río Gallegos", "lat": -51.6230, "lon": -69.2168},
    ]
    for c in ciudades_argentinas:
        cx, cy = m(c['lon'], c['lat'])
        m.plot(cx, cy, marker='o', color='black', markersize=4, zorder=5)
        plt.text(cx + 5000, cy + 5000, c['nombre'], fontsize=8, ha='right', va='bottom')

    plt.text(0.15, 0.9, "Argentina", transform=ax.transAxes, fontsize=16, fontweight='bold', color='black',
             ha='center', va='center', alpha=0.5)

    ax_inlet = fig.add_axes([0.62, 0.065, 0.22, 0.22])
    m_inlet = Basemap(projection='cyl', resolution='c', ax=ax_inlet)
    m_inlet.drawcoastlines(linewidth=0.5); m_inlet.drawcountries(linewidth=0.5)
    m_inlet.drawmapboundary(fill_color='lightblue'); m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')
    rect_lons = [-70, -45, -45, -70, -70]; rect_lats = [-55, -55, -35, -35, -55]
    m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5)

    ax.set_title("N geodesic radial lines from H10N for TL modeling", fontsize=18)
    ax.legend(loc='upper right')

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    fig.savefig(salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return salida

if __name__ == "__main__":
    os.makedirs("figuras", exist_ok=True)
    archivos_csv = sorted(glob.glob("input-data/*.csv"))
    print(f"Procesando {len(archivos_csv)} archivos en paralelo...")

    # Mapas interpolados por archivo (con TL saturado en 200)
    if archivos_csv:
        with ProcessPoolExecutor() as executor:
            resultados = list(executor.map(procesar_archivo, archivos_csv))
    else:
        resultados = []

    print("\n=== Resultados ===")
    for r in resultados:
        print(r)

    # ÚNICO mapa de Data points (ubicaciones iguales)
    try:
        if archivos_csv:
            out_png = crear_mapa_datapoints_desde_csv(archivos_csv[0], salida="figuras/mapa_datapoints_unico.png")
            print("Mapa de Data points:", out_png)
        else:
            print("No hay CSVs para crear el mapa de Data points.")
    except Exception as e:
        print("No se pudo crear el mapa de Data points:", e)



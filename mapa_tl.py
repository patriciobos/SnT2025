import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import alphashape
from shapely.geometry import Point

from fastkml import kml
from shapely.geometry import Polygon, LineString

def superponer_kml(ax, kml_file, color='red'):
    from fastkml import kml
    from shapely.geometry import mapping
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.geometry.polygon import Polygon

    with open(kml_file, 'rt', encoding='utf-8') as f:
        doc = f.read()

    k = kml.KML()
    k.from_string(doc.encode('utf-8'))

    def iter_features(features):
        for feature in features:
            if hasattr(feature, 'geometry') and feature.geometry is not None:
                geom = feature.geometry
                if isinstance(geom, (Polygon, MultiPolygon)):
                    gjson = mapping(geom)
                    if gjson['type'] == 'Polygon':
                        coords = gjson['coordinates'][0]
                        x, y = zip(*coords)
                        ax.plot(x, y, color=color, linewidth=2)
                    elif gjson['type'] == 'MultiPolygon':
                        for poly in gjson['coordinates']:
                            coords = poly[0]
                            x, y = zip(*coords)
                            ax.plot(x, y, color=color, linewidth=2)
            # Recur en sub-features si las hay
            if hasattr(feature, 'features'):
                iter_features(feature.features())

    iter_features(k.features())

def procesar_archivo(filename):
    basename = os.path.basename(filename)

    match = re.search(r"f([\d.]+)\s*Hz", basename)
    frecuencia_str = f"{match.group(1)} Hz" if match else "Frecuencia desconocida"
    frecuencia_limpia = match.group(1).replace('.', '_') if match else "desconocida"
    nombre_figura = f"mapa_tl_z_8_f{frecuencia_limpia}"

    print(f"Procesando: {basename}")

    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip().str.lower()

        columnas_TL = ['tl_z_8']
        for col in ['lat', 'lon', 'bat'] + columnas_TL:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
        df['lat'] = df['lat'].astype(float)
        df['lon'] = df['lon'].astype(float)
        df['bat'] = -df['bat'].astype(float)

        TLmax = 210
        col_TL = columnas_TL[0]
        df_filtrado = df[df[col_TL] < TLmax].copy()
        lat_f = df_filtrado['lat'].values
        lon_f = df_filtrado['lon'].values
        TL = df_filtrado[col_TL].values

        grid_lat = np.linspace(np.min(lat_f), np.max(lat_f), 300)
        grid_lon = np.linspace(np.min(lon_f), np.max(lon_f), 300)
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

        points = np.column_stack((lon_f, lat_f))
        grid_TL = griddata(points, TL, (grid_lon2d, grid_lat2d), method='linear')

        alpha = 3.0
        concave_hull = alphashape.alphashape(points, alpha)
        mask = np.array([
            concave_hull.contains(Point(x, y))
            for x, y in zip(grid_lon2d.ravel(), grid_lat2d.ravel())
        ])
        grid_TL_masked = np.full_like(grid_TL, np.nan)
        grid_TL_masked.ravel()[mask] = grid_TL.ravel()[mask]

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

        x, y = m(grid_lon2d, grid_lat2d)
        norm = Normalize(vmin=np.nanmin(grid_TL_masked), vmax=np.nanmax(grid_TL_masked))
        cmap = colormaps['viridis']
        im = m.pcolormesh(x, y, grid_TL_masked, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(im, ax=ax_main, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label(f"{col_TL} (dB)")

        # Ciudades argentinas
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

        # Coordenadas objetivo
        coordenadas_objetivo = [
            {"lat": -38.5092, "lon": -56.4850, "nombre": "MDQ"},
            {"lat": -44.9512, "lon": -63.8894, "nombre": "GSJ"},
            {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},  
        ]
        for punto in coordenadas_objetivo:
            px, py = m(punto["lon"], punto["lat"])
            m.plot(px, py, marker='*', color='red', markersize=8, zorder=6)
            plt.text(px + 5000, py + 5000, punto["nombre"], fontsize=9, ha='left', va='bottom', color='red')

        # Etiqueta "Argentina"
        plt.text(0.15, 0.9, "Argentina", transform=ax_main.transAxes,
                 fontsize=16, fontweight='bold', color='black',
                 ha='center', va='center', alpha=0.5)

        # Rectángulo ZAIS
        #x1, y1 = m(-57.3333, -38.5833)
        #x2, y2 = m(-55.55, -38.0)
        #m.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='black', linestyle='--', linewidth=1.5)

        # Inlet planisferio
        ax_inlet = fig.add_axes([0.55, 0.1, 0.22, 0.22])
        m_inlet = Basemap(projection='cyl', resolution='c', ax=ax_inlet)
        m_inlet.drawcoastlines(linewidth=0.5)
        m_inlet.drawcountries(linewidth=0.5)
        m_inlet.drawmapboundary(fill_color='lightblue')
        m_inlet.fillcontinents(color='lightgray', lake_color='lightblue')

        ll_lon, ll_lat = -70, -55
        ur_lon, ur_lat = -45, -35
        rect_lons = [ll_lon, ur_lon, ur_lon, ll_lon, ll_lon]
        rect_lats = [ll_lat, ll_lat, ur_lat, ur_lat, ll_lat]
        m_inlet.plot(rect_lons, rect_lats, color='red', linewidth=1.5)

        # Superponer archivo KML si existe
        ruta_kml = "Ronda_1.kml"
        if os.path.exists(ruta_kml):
            superponer_kml(ax_main, ruta_kml, color='red')


        ax_main.set_title(f"Transmission Loss from H10N @{frecuencia_str}, z = 8 m.", fontsize=14)
        os.makedirs("output-data", exist_ok=True)
        fig.savefig(f"output-data/{nombre_figura}_basemap.png", dpi=300, bbox_inches='tight')
        plt.close()
        return f"{basename} OK"
    except Exception as e:
        return f"{basename} ERROR: {e}"

if __name__ == "__main__":
    archivos_csv = sorted(glob.glob("input-data/pato-vienna-f80.0 Hz.csv"))
    resultados = [procesar_archivo(f) for f in archivos_csv]

    print("\n=== Resultados ===")
    for r in resultados:
        print(r)


# === Estándar de Python ===
import os
import re
from datetime import datetime

# === Terceros: NumPy, Pandas, SciPy ===
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay

# === Visualización: Matplotlib y Basemap ===
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap

# === Mapas interactivos: Folium y Branca ===
import folium
import branca.colormap as brc_cm

# === Otros: Alphashape, Shapely, Xarray ===
import alphashape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import xarray as xr

# === Cargar archivo ===
filename = "results/pato-vienna-f50.0 Hz.csv"
basename = os.path.basename(filename)

match = re.search(r"f([\d.]+)\s*Hz", basename)
frecuencia_str = f"{match.group(1)} Hz" if match else "Frecuencia desconocida"

df = pd.read_csv(filename)
df.columns = df.columns.str.strip().str.lower()

# === Validar columnas ===
columnas_TL = ['tl_z_8']
nombres_figura = [f"mapa_tl_z_8 a {frecuencia_str}"]

for col in ['lat', 'lon', 'bat'] + columnas_TL:
    df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
df['lat'] = df['lat'].astype(float)
df['lon'] = df['lon'].astype(float)
df['bat'] = -df['bat'].astype(float)

TLmax = 210  # Valor máximo de TL para filtrar

# === Salida ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = f"mapas_tl_basemap_{timestamp}.pdf"

with PdfPages(pdf_filename) as pdf:
    for i, col_TL in enumerate(columnas_TL):
        df_filtrado = df[df[col_TL] < TLmax].copy()
        lat_f = df_filtrado['lat'].values
        lon_f = df_filtrado['lon'].values
        TL = df_filtrado[col_TL].values

        # === Interpolación sobre grilla regular ===
        grid_lat = np.linspace(np.min(np.asarray(lat_f)), np.max(np.asarray(lat_f)), 300)
        grid_lon = np.linspace(np.min(np.asarray(lon_f)), np.max(np.asarray(lon_f)), 300)
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

        lon_f = np.asarray(lon_f).flatten()
        lat_f = np.asarray(lat_f).flatten()
        points = np.column_stack((lon_f, lat_f))
        grid_TL = griddata(points, TL, (grid_lon2d, grid_lat2d), method='linear')

        # === Alpha shape (concave hull) ===
        alpha = 3.0  # menor → más ajustado, mayor → más cercano al convex hull
        concave_hull = alphashape.alphashape(points, alpha)

        # === Crear máscara con los puntos dentro del polígono ===
        mask = np.array([
            concave_hull.contains(Point(x, y)) 
            for x, y in zip(grid_lon2d.ravel(), grid_lat2d.ravel())
        ])

        grid_TL_masked = np.full_like(grid_TL, np.nan)
        grid_TL_masked.ravel()[mask] = grid_TL.ravel()[mask]

        # === Basemap ===
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
        m.drawparallels(range(-55, -34, 5), labels=[1,0,0,0])
        m.drawmeridians(range(-70, -44, 5), labels=[0,0,0,1])

        # === Transformar grilla a proyección ===
        x, y = m(grid_lon2d, grid_lat2d)

        # === Mostrar interpolación ===
        norm = Normalize(
            vmin=np.nanmin(np.asarray(grid_TL_masked)),
            vmax=np.nanmax(np.asarray(grid_TL_masked))
        )
        cmap = mpl_cm.get_cmap('viridis')
        im = m.pcolormesh(x, y, grid_TL_masked, cmap=cmap, norm=norm, shading='auto')

        cbar = plt.colorbar(im, ax=ax_main, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label(f"{col_TL} (dB)")

        # === Ciudades ===
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

        # === Rectángulo ZAIS ===
        zais_bounds = [[-38.5833, -57.3333], [-38.0, -55.55]]
        lats_zais = [zais_bounds[0][0], zais_bounds[1][0]]
        lons_zais = [zais_bounds[0][1], zais_bounds[1][1]]
        x1, y1 = m(lons_zais[0], lats_zais[0])
        x2, y2 = m(lons_zais[1], lats_zais[1])
        x1 = float(np.squeeze(x1))
        x2 = float(np.squeeze(x2))
        y1 = float(np.squeeze(y1))
        y2 = float(np.squeeze(y2))

        m.plot([x1, x2, x2, x1, x1],
               [y1, y1, y2, y2, y1],
               color='black', linestyle='--', linewidth=1.5)
        plt.text((x1 + x2) / 2, y1 - 100000, "ZAIS", ha='center', va='top', fontsize=9)

        # === Título y salida ===
        ax_main.set_title(f"Mapa de {col_TL} - {frecuencia_str}", fontsize=14)
        fig.savefig(f"{nombres_figura[i]}_basemap.png", dpi=300, bbox_inches='tight')
        plt.close()

        #########################
        # === Folium ===
        #########################

        TL = np.array(TL, dtype=float)

        # Crear mapa Folium centrado en la región
        center_lat = np.mean(df['lat'].to_numpy())
        center_lon = np.mean(df['lon'].to_numpy())
        m_folium = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')

        # Colormap normalizado según TL
        min_TL = np.min(TL)
        max_TL = np.max(TL)
        colormap = brc_cm.linear.viridis.scale(min_TL, max_TL)
        colormap.caption = f"{col_TL} (dB)"
        m_folium.add_child(colormap)

        # Agregar puntos
        for lt, ln, val in zip(lat_f, lon_f, TL):
            folium.CircleMarker(
                location=[lt, ln],
                radius=3,
                fill=True,
                fill_opacity=0.7,
                color=colormap(val),
                weight=0,
                tooltip=f"TL: {val:.1f} dB<br>Bat: {df.loc[(df['lat'] == lt) & (df['lon'] == ln), 'bat'].values[0]:.1f} m<br>Lat: {lt:.4f}<br>Lon: {ln:.4f}"
            ).add_to(m_folium)

        # Guardar HTML
        folium_filename = f"{nombres_figura[i]}.html"
        m_folium.save(folium_filename)

print("Mapas guardados en formato PNG.")

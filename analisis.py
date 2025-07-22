import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
from datetime import datetime
import re
import os

import folium
import branca.colormap as cm


# Nombre del archivo
filename = "results/pato-vienna-f50.0 Hz.csv"
basename = os.path.basename(filename)

# Extraer frecuencia del nombre
match = re.search(r"f([\d.]+)\s*Hz", basename)
if match:
    frecuencia = match.group(1)
    frecuencia_str = f"{frecuencia} Hz"
else:
    frecuencia_str = "Frecuencia desconocida"

# Leer CSV
df = pd.read_csv(filename)
df.columns = df.columns.str.strip().str.lower()  # por las dudas

# Columnas de interés
columnas_TL = ['tl_z_8', 'tl_z_half', 'tl_max_z']
nombres_figura = [
    f"mapa_tl_z_8 a {frecuencia_str}",
    f"mapa_tl_z_half a {frecuencia_str}",
    f"mapa_tl_max_z a {frecuencia_str}"
]

# Validar y convertir columnas necesarias
for col in ['lat', 'lon', 'bat'] + columnas_TL:
    df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
df['lat'] = df['lat'].astype(float)
df['lon'] = df['lon'].astype(float)
df['bat'] = -df['bat'].astype(float)

# Cargar batimetría GEBCO 2024
ds = xr.open_dataset("gebco_2024_n-35.0_s-55.0_w-70.0_e-50.0.nc")
elev = ds["elevation"].values
lons = ds["lon"].values
lats = ds["lat"].values
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Generar nombre de PDF
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = f"mapas_tl_z_all_{timestamp}.pdf"

# Crear PDF con los 3 mapas
with PdfPages(pdf_filename) as pdf:
    for i, col_TL in enumerate(columnas_TL):
        # Filtrar por TL < 200
        df_filtrado = df[df[col_TL] < 180].copy()
        lat_f = df_filtrado['lat'].values
        lon_f = df_filtrado['lon'].values
        TL = df_filtrado[col_TL].values

        # Crear figura y ejes
        fig = plt.figure(figsize=(10, 9))
        ax_main = fig.add_axes((0.1, 0.1, 0.85, 0.85))

        # Crear mapa base
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

        # Proyectar coordenadas y graficar TL
        x, y = m(lon_f, lat_f)
        sc = m.scatter(x, y, c=TL, s=1, cmap='viridis', edgecolors='none', zorder=5)
        cbar = plt.colorbar(sc, ax=ax_main, orientation='vertical', shrink=0.6, pad=0.02)
        cbar.set_label(f"{col_TL} (dB)")

        # Curvas de batimetría
        grid_x, grid_y = m(lon_grid, lat_grid)
        contours = m.contour(grid_x, grid_y, elev,
                            levels=[-2000, -1000, -500, -200, -100, -50],
                            colors='black', linewidths=0.4, linestyles='dashed')
        plt.clabel(contours, inline=True, fontsize=9, fmt='%d m', inline_spacing=5, rightside_up=True)

        # Agregar nombres de país y ciudades al mapa base
        ciudades = {
            "ARGENTINA": (-66.0, -36.0),
            "Ushuaia": (-68.3, -54.8),
            "Comodoro Rivadavia": (-67.48, -45.87),
            "Puerto Madryn": (-66, -42.77),
            "Bahía Blanca": (-62.27, -38.72),
            "Mar del Plata": (-57.5426, -38.0055),
            "Río Gallegos": (-69.2167, -51.6333),
            "Puerto Argentino": (-57.85, -51.7)
        }
     

        for nombre, (lon_c, lat_c) in ciudades.items():
            x_c, y_c = m(lon_c, lat_c)
            ax_main.text(x_c, y_c, nombre,
                         fontsize=8, ha='center', va='bottom',
                         bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Inset map
        ax_inset = fig.add_axes((0.73, 0.05, 0.24, 0.24))
        m_inset = Basemap(projection='cyl',
                          llcrnrlat=-90, urcrnrlat=90,
                          llcrnrlon=-180, urcrnrlon=180,
                          resolution='c', ax=ax_inset)

        m_inset.drawcoastlines()
        m_inset.drawcountries()
        m_inset.drawmapboundary(fill_color='lightblue')
        m_inset.fillcontinents(color='lightgray', lake_color='lightblue')
        m_inset.plot([-70, -45, -45, -70, -70], [-55, -55, -35, -35, -55], color='blue', linewidth=2)

        # Título
        ax_main.set_title(f"Mapa de {col_TL} - {frecuencia_str}", fontsize=14)

        #########################
        # folium
        #########################

                # Crear mapa Folium centrado en la región
        center_lat = np.mean(lat_f)
        center_lon = np.mean(lon_f)
        m_folium = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')

        # Colormap normalizado según TL
        min_TL = np.min(TL)
        max_TL = np.max(TL)
        colormap = cm.linear.viridis.scale(min_TL, max_TL)

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
                tooltip=f"TL: {val:.1f} dB<br>Bat: {df.loc[(df['lat'] == lt) & (df['lon'] == ln), 'bat'].values[0]:.1f} m"
            ).add_to(m_folium)


        # Guardar HTML
        folium_filename = f"{nombres_figura[i]}.html"
        m_folium.save(folium_filename)

        # Guardar
        pdf.savefig(fig)
        fig.savefig(f"{nombres_figura[i]}.png", dpi=600, bbox_inches='tight')
        plt.close()

print(f"PDF generado: {pdf_filename}")

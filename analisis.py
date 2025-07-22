import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages

# Leer datos
df = pd.read_csv("data.csv", header=0)

# Convertir columnas necesarias
columnas_TL = ['tl_z_8', 'tl_z_half', 'tl_max_z']
nombres_figura = ['mapa_tl_z_8', 'mapa_tl_z_half', 'mapa_tl_max_z']

# Asegurar valores válidos
for col in ['latitud', 'longitud', 'bat'] + columnas_TL:
    df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
df['latitud'] = df['latitud'].astype(float)
df['longitud'] = df['longitud'].astype(float)
df['bat'] = -df['bat'].astype(float)  # invertir signo para batimetría

# Extraer info común
lat = df['latitud'].values
lon = df['longitud'].values
bat = df['bat'].values

# Interpolación batimétrica para reutilizar
grid_lon = np.linspace(min(lon), max(lon), 300)
grid_lat = np.linspace(min(lat), max(lat), 300)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
grid_bat = griddata((lon, lat), bat, (grid_lon, grid_lat), method='linear')

# Imprimir máximos y valores inválidos
print("Máximo tl_z_8m:", df['tl_z_8'].max())
print("Máximo tl_z_half:", df['tl_z_half'].max())
print("Máximo tl_max_z:", df['tl_max_z'].max())

# Crear PDF
with PdfPages("mapas_tl_z_all.pdf") as pdf:
    for i, col_TL in enumerate(columnas_TL):
        # Filtrar solo los datos con TL < 200 en esta profundidad
        df_filtrado = df[pd.to_numeric(df[col_TL], errors='coerce') < 200].copy()

        # Extraer coordenadas y TL de esta capa
        lat_f = df_filtrado['latitud'].astype(float).values
        lon_f = df_filtrado['longitud'].astype(float).values
        TL = df_filtrado[col_TL].astype(float).values

        fig = plt.figure(figsize=(10, 9))
        ax_main = fig.add_axes((0.1, 0.1, 0.85, 0.85))

        # Crear mapa
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

        # Proyección de coordenadas
        x, y = m(lon_f, lat_f)
        
        # Dibujar TL
        sc = m.scatter(x, y, c=TL, s=1, cmap='viridis', edgecolors='none', zorder=5)
        cbar = plt.colorbar(sc, ax=ax_main, orientation='vertical', shrink=0.6, pad=0.02)
        cbar.set_label(f"{col_TL} (dB)")

        # Curvas de batimetría
        grid_x, grid_y = m(grid_lon, grid_lat)
        contours = m.contour(grid_x, grid_y, grid_bat,
                            levels=[-1000, -500, -200, -100, -50],
                            colors='black', linewidths=0.6, linestyles='dashed')
        plt.clabel(contours, inline=True, fontsize=9, fmt='%d m', inline_spacing=5, rightside_up=True)

        # Inset
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

        # Guardar como página en PDF y como PNG individual si se desea
        pdf.savefig(fig)
        fig.savefig(f"{nombres_figura[i]}.png", dpi=600, bbox_inches='tight')
        plt.close()

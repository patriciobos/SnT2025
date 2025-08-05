import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from multiprocessing import Pool, cpu_count

# ========== CONFIGURACIÓN ==========
ZONA = 'gsj'  # Elegir entre 'zais', 'arasj', 'gsj'
VARIABLE = 'tl_z_8'  # Elegir entre 'tl_z_8', 'tl_z_half', 'tl_z_max'
CARPETA = 'input-platform'
SALIDA = 'figuras'

CIUDADES = {
    "Buenos Aires": (-58.417, -34.6),
    "Mar del Plata": (-57.55, -38.0),
    "Bahía Blanca": (-62.27, -38.72),
    "Puerto Madryn": (-65.04, -42.77),
    "Rawson": (-65.1, -43.3),
    "Comodoro Rivadavia": (-67.5, -45.87),
    "Río Gallegos": (-69.22, -51.63),
    "Ushuaia": (-68.3, -54.8)
}

# ========== FUNCIONES ==========
def procesar_archivo(archivo):
    try:
        if not archivo.endswith('.csv') or f'platform-{ZONA}-' not in archivo:
            return f"IGNORADO: {archivo}"

        df = pd.read_csv(os.path.join(CARPETA, archivo))
        if not {'lat', 'lon', VARIABLE}.issubset(df.columns):
            return f"ERROR columnas: {archivo}"

        df = df[df[VARIABLE] <= 210]  # Filtrar valores mayores a 210

        lons = df['lon'].values
        lats = df['lat'].values
        vals = df[VARIABLE].values

        if len(vals) < 3:
            return f"ERROR pocos puntos: {archivo}"

        fig = plt.figure(figsize=(10, 9))
        m = Basemap(projection='merc',
                    llcrnrlat=-55, urcrnrlat=-35,
                    llcrnrlon=-70, urcrnrlon=-45,
                    resolution='i')

        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawparallels(np.arange(-60, -30, 5), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-75, -40, 5), labels=[0, 0, 0, 1])

        for nombre, (lon, lat) in CIUDADES.items():
            x, y = m(lon, lat)
            plt.plot(x, y, 'ko', markersize=4)
            plt.text(x + 10000, y + 10000, nombre, fontsize=8)

        x, y = m(lons, lats)
        sc = m.scatter(x, y, c=vals, cmap='viridis', edgecolors='k', s=40, norm=Normalize(vmin=min(vals), vmax=max(vals)))
        cbar = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label(f"{VARIABLE} (dB)")

        frecuencia = re.findall(r"f([0-9.]+)", archivo)
        titulo = f"Zona {ZONA.upper()} - Frecuencia {frecuencia[0]} Hz - Variable {VARIABLE}" if frecuencia else f"Zona {ZONA.upper()} - Variable {VARIABLE}"
        plt.title(titulo)

        os.makedirs(SALIDA, exist_ok=True)
        nombre_salida = f"{SALIDA}/mapa_{ZONA}_{frecuencia[0].replace('.', '_')}Hz_{VARIABLE}.png"
        plt.savefig(nombre_salida, dpi=300, bbox_inches='tight')
        plt.close()
        return f"OK: {archivo}"

    except Exception as e:
        return f"ERROR: {archivo} → {e}"

# ========== EJECUCIÓN ==========
if __name__ == '__main__':
    archivos = [f for f in os.listdir(CARPETA) if f.endswith('.csv') and f'platform-{ZONA}-' in f]
    with Pool(cpu_count()) as pool:
        resultados = pool.map(procesar_archivo, archivos)
    for r in resultados:
        print(r)

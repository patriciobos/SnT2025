import os
import pandas as pd
import numpy as np
from pathlib import Path
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time


# === Configuración general ===
carpeta_data = Path("input-data")
carpeta_output = Path("output-data")
columnas_tl = ["tl_z_8", "tl_z_half", "tl_max_z"]
columnas_coord = ["lat", "lon"]

# Coordenadas objetivo con campo opcional "nombre"
coordenadas_objetivo = [
    {"lat": -38.5092, "lon": -56.4850, "nombre": "ZAIS"},
    {"lat": -44.9512, "lon": -63.8894, "nombre": "GSJ"},
    {"lat": -45.9501, "lon": -59.7736, "nombre": "ARASJ"},  
]

# === Detectar frecuencias disponibles (solo una vez, compartido por todos) ===
archivos_csv = sorted(Path(carpeta_data).glob("*.csv"))
frecuencias_disponibles = []
for archivo in archivos_csv:
    nombre = archivo.stem
    if "Hz" in nombre:
        try:
            f = float(nombre.split("f")[-1].replace("Hz", "").strip())
            frecuencias_disponibles.append((f, archivo))
        except ValueError:
            continue
frecuencias_disponibles.sort()

# === Función de utilidad ===
def encontrar_fila_mas_cercana(df, lat0, lon0):
    distancias = df.apply(
        lambda row: geodesic((lat0, lon0), (row[columnas_coord[0]], row[columnas_coord[1]])).meters,
        axis=1
    )
    idx_min = distancias.idxmin()
    return df.loc[idx_min]

# === Función paralelizable por punto ===
def procesar_punto(punto):
    lat_obj = punto["lat"]
    lon_obj = punto["lon"]
    nombre_amigable = punto.get("nombre", f"{lat_obj:.4f}_{lon_obj:.4f}").replace(".", "p")

    filas_resultado = []

    for f, archivo in frecuencias_disponibles:
        df = pd.read_csv(archivo)
        if not set(columnas_coord + columnas_tl).issubset(df.columns):
            continue

        fila_cercana = encontrar_fila_mas_cercana(df, lat_obj, lon_obj)
        fila_resultado = {"frecuencias": f}
        for col in columnas_tl:
            fila_resultado[col] = fila_cercana[col]
        filas_resultado.append(fila_resultado)

    df_resultado = pd.DataFrame(filas_resultado).sort_values("frecuencias")
    nombre_archivo = f"TL_{nombre_amigable}.csv"
    carpeta_output.mkdir(parents=True, exist_ok=True)  # asegurar que existe
    df_resultado.to_csv(carpeta_output / nombre_archivo, index=False)

    print(f"Archivo guardado: {carpeta_output / nombre_archivo}")


# === Ejecutar en paralelo ===
if __name__ == "__main__":
    
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        if not archivos_csv:
            print("No se encontraron archivos CSV en la carpeta de entrada.")
            exit(1)
        executor.map(procesar_punto, coordenadas_objetivo)

    elapsed_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
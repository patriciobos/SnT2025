#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ======== CONFIG =========
# =========================
ZONA = "arasj"          # "gsj" | "arasj" | "zais"
FRECUENCIA_HZ = 100.0      # p. ej. 2.0
CARPETA_INPUT = "input-platform"
CARPETA_OUTPUT = "mapas"
TL_COL = "tl_z_8"

# Visual
BINS = 40                # o poné None para calcularlo con Freedman–Diaconis
TL_MIN = None            # p. ej. 40.0 (opcional)
TL_MAX = None            # p. ej. 180.0 (opcional)
DPI = 200
MOSTRAR = True           # True para ver la ventana, False sólo guarda PNG

# Matching
FREQ_TOL = 0.01          # tolerancia al buscar f=X Hz en el nombre


# =========================
# ====== UTILIDADES =======
# =========================
def buscar_archivo(carpeta, zona, freq_obj, tol=0.01):
    """
    Busca un archivo con patrón:
    pato-vienna-platform-<zona>-f<freq> Hz.csv
    Devuelve ruta o None.
    """
    zona = zona.lower()
    patron = re.compile(
        rf"^pato-vienna-platform-{zona}-f(\d+(?:\.\d+)?)\s*Hz\.csv$",
        re.IGNORECASE
    )
    candidatos = []
    try:
        for fname in os.listdir(carpeta):
            m = patron.match(fname)
            if m:
                f = float(m.group(1))
                if abs(f - freq_obj) <= tol:
                    candidatos.append((abs(f - freq_obj), f, os.path.join(carpeta, fname)))
    except FileNotFoundError:
        return None

    if not candidatos:
        return None
    candidatos.sort(key=lambda x: x[0])
    return candidatos[0][2]

def extraer_frecuencia_desde_nombre(nombre):
    m = re.search(r"f(\d+(?:\.\d+)?)\s*Hz", nombre)
    return float(m.group(1)) if m else None

def bins_freedman_diaconis(data):
    # número de bins recomendado si querés automático
    data = np.asarray(data)
    if data.size < 2:
        return 10
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10
    h = 2 * iqr * (data.size ** (-1/3))
    if h <= 0:
        return 10
    nbins = int(np.ceil((data.max() - data.min()) / h))
    return max(nbins, 5)


# =========================
# ========= MAIN ==========
# =========================
def main():
    ruta = buscar_archivo(CARPETA_INPUT, ZONA, FRECUENCIA_HZ, tol=FREQ_TOL)
    if ruta is None:
        print(f"❌ No se encontró archivo en '{CARPETA_INPUT}' para zona '{ZONA}' "
              f"con frecuencia ~ {FRECUENCIA_HZ} Hz (±{FREQ_TOL}).")
        return

    try:
        df = pd.read_csv(ruta)
    except Exception as e:
        print(f"❌ Error leyendo CSV '{ruta}': {e}")
        return

    if TL_COL not in df.columns:
        print(f"❌ La columna '{TL_COL}' no está en el archivo. "
              f"Columnas disponibles: {list(df.columns)}")
        return

    tl = pd.to_numeric(df[TL_COL], errors="coerce").dropna()

    if TL_MIN is not None:
        tl = tl[tl >= TL_MIN]
    if TL_MAX is not None:
        tl = tl[tl <= TL_MAX]

    if tl.empty:
        print("⚠️ No hay datos válidos para graficar (tras filtros opcionales).")
        return

    f_real = extraer_frecuencia_desde_nombre(os.path.basename(ruta)) or FRECUENCIA_HZ
    zona_up = ZONA.upper()

    nbins = BINS if BINS is not None else bins_freedman_diaconis(tl.values)

    plt.figure(figsize=(8,5))
    plt.hist(tl.values, bins=nbins, alpha=0.85, edgecolor='k')
    plt.xlabel(f"TL (dB) — {TL_COL}")
    plt.ylabel("Recuentos")
    plt.title(f"Histograma TL — {zona_up} — f={f_real} Hz")

    mu, med, std = np.mean(tl), np.median(tl), np.std(tl)
    txt = f"n={len(tl)}\nμ={mu:.1f} dB\nmediana={med:.1f} dB\nσ={std:.1f} dB"
    plt.gca().annotate(txt, xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    out_png = os.path.join(CARPETA_OUTPUT, f"hist_{ZONA}_f{f_real}Hz_{TL_COL}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    print(f"✅ Histograma guardado en: {out_png}")

    if MOSTRAR:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()

# Generación de Mapas TL con Basemap

Este proyecto permite generar mapas de niveles de transmisión (TL) a partir de archivos `.csv` con datos geográficos e interpolarlos sobre mapas utilizando **Basemap** y **paralelización con múltiples núcleos**.

---

## 📁 Estructura esperada
.
├── input-data/ # Carpeta con archivos CSV de entrada (por ejemplo, pato-vienna-f50.0 Hz.csv)
├── figuras/ # Carpeta de salida donde se guardarán las figuras generadas
├── analisis.py # Script principal que genera los mapas
├── requirements.txt # Lista de paquetes necesarios
└── README.md
---

## ⚙️ Requisitos

### 1. Python

Se recomienda Python **3.9 o superior**.

### 2. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

3. Instalar dependencias

pip install -r requirements.txt

    ⚠️ Nota sobre Basemap:
    El paquete basemap está deprecado y puede requerir la instalación de dependencias del sistema.
    Se recomienda usar entornos como Conda si hay problemas con pip.
    Alternativa moderna recomendada: Cartopy.
    
    


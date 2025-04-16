# HAM10000 Skin Lesion Classification using Vision Transformers

Este repositorio contiene una implementación de un modelo de clasificación de lesiones cutáneas basado en Vision Transformers, desarrollado como parte de una tesis de maestría. El proyecto utiliza el conjunto de datos HAM10000, que contiene imágenes dermatoscópicas clasificadas en 7 categorías diferentes de lesiones cutáneas.

## Autor
**Nicolas A. Gentile**  
Email: nag.gentile@gmail.com

## Descripción del Proyecto

Este proyecto implementa una solución para la clasificación automática de lesiones cutáneas utilizando Vision Transformers (ViT), un tipo de arquitectura de redes neuronales que ha demostrado gran efectividad en tareas de visión por computadora. El enfoque no sólo utiliza características visuales extraídas de las imágenes, sino que también incorpora datos clínicos adicionales como la edad del paciente, el sexo y la localización de la lesión para mejorar la precisión diagnóstica.

### Conjunto de Datos

El proyecto utiliza el conjunto de datos HAM10000 (Human Against Machine with 10000 training images), que contiene imágenes dermatoscópicas de lesiones cutáneas clasificadas en 7 categorías:

1. `nv`: Nevos melanocíticos (Melanocytic nevi)
2. `mel`: Melanoma (Melanoma)
3. `bkl`: Lesiones tipo queratosis benignas (Benign keratosis-like lesions)
4. `bcc`: Carcinoma basocelular (Basal cell carcinoma)
5. `akiec`: Queratosis actínica (Actinic keratoses)
6. `vasc`: Lesiones vasculares (Vascular lesions)
7. `df`: Dermatofibroma (Dermatofibroma)

El dataset HAM10000 fue publicado por Tschandl et al. y está disponible para su descarga en Harvard Dataverse:

[![DOI](https://img.shields.io/badge/DOI-10.7910%2FDVN%2FDBW86T-blue)](https://doi.org/10.7910/DVN/DBW86T)

[HAM10000 Dataset - Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

El dataset contiene:
- `HAM10000_metadata.csv`: Archivo con metadatos de cada imagen, incluyendo diagnóstico, edad, sexo y localización.
- `HAM10000_images_part1.zip` y `HAM10000_images_part2.zip`: Archivos comprimidos que contienen todas las imágenes en formato .jpg.

## Estructura del Repositorio

- **scripts/**: Contiene los scripts principales para entrenar y evaluar el modelo.
  - `run_training.py`: Script principal para entrenamiento y evaluación del modelo.
- **classes/**: Implementaciones de clases para el manejo de datos y transformaciones.
  - `dataset.py`: Implementación del dataset personalizado para HAM10000.
  - `transforms.py`: Transformaciones aplicadas a las imágenes.
- **models/**: Contiene definiciones y pesos de modelos.
  - `vision_transformer.py`: Implementación del Vision Transformer (ViT).
  - `final_model.pth`: Modelo entrenado final.
- **settings/**: Archivos de configuración y parámetros.
  - `config.yaml`: Archivo de configuración principal con hiperparámetros.
  - `parameters.py`: Definición de rutas y parámetros globales.
- **data/**: Directorio para almacenar el conjunto de datos.
  - `HAM10000_metadata.csv`: Archivo de metadatos del dataset.
  - `HAM10000_images/`: Directorio que contiene todas las imágenes del dataset.
- **evaluate/**: Scripts para la evaluación del modelo.
- **fit/**: Funciones para entrenamiento y validación.
- **runs/**: Registros de entrenamientos para TensorBoard.
- **logs/**: Archivos de registro del entrenamiento.

## Características Principales

- **Arquitectura basada en Vision Transformers (ViT)**: Utiliza modelos preentrenados de ViT adaptados para la clasificación de lesiones cutáneas.
- **Integración de características clínicas**: Incorpora datos de edad, sexo y localización de la lesión como características adicionales.
- **Optimización de hiperparámetros**: Implementa búsqueda automática de hiperparámetros para maximizar el rendimiento del modelo.
- **Aumento de datos**: Utiliza técnicas de aumento de datos para mejorar la generalización del modelo.
- **Visualización de resultados**: Integración con TensorBoard para monitoreo y visualización del entrenamiento.

## Requisitos

```
pandas
pillow
matplotlib
seaborn
pyyaml
optuna
torch
torchvision
timm
scikit-learn
tensorboard
```

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd Tesis_HAM10000
```

2. Crear y activar un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Descargar el conjunto de datos HAM10000:
   - Acceder a [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
   - Descargar `HAM10000_metadata.csv`, `HAM10000_images_part1.zip` y `HAM10000_images_part2.zip`
   - Crear un directorio `data` en la raíz del proyecto si no existe
   - Colocar `HAM10000_metadata.csv` en el directorio `data/`
   - Extraer todas las imágenes de los archivos .zip en un directorio `data/HAM10000_images/`

La estructura de directorios de datos debe quedar así:
```
data/
├── HAM10000_metadata.csv
└── HAM10000_images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ... (todas las imágenes)
```

**IMPORTANTE**: El proyecto requiere que el dataset HAM10000 esté descargado y correctamente estructurado en la carpeta `data` para funcionar.

## Uso

### Entrenamiento del Modelo

Para entrenar el modelo con los parámetros configurados en `settings/config.yaml`:

```bash
python scripts/run_training.py
```

### Configuración

La configuración principal del modelo y el entrenamiento se realiza a través del archivo `settings/config.yaml`. Los principales parámetros que se pueden ajustar incluyen:

- **Tamaño de lote (batch_size)**: Controla la cantidad de imágenes procesadas en cada iteración.
- **Tasa de aprendizaje (learning_rate)**: Controla la velocidad de actualización de los pesos del modelo.
- **Épocas (epochs)**: Número de veces que el modelo recorre todo el conjunto de datos durante el entrenamiento.
- **Configuración de aumentación de datos**: Transformaciones aplicadas a las imágenes para aumentar la variabilidad.
- **Uso de características adicionales**: Activar/desactivar el uso de datos clínicos (edad, sexo, localización).

### Resultados

El modelo entrenado puede alcanzar una precisión superior al 85% en el conjunto de prueba, lo que demuestra su efectividad para la clasificación de lesiones cutáneas. Los resultados detallados y las métricas de evaluación se almacenan en los directorios `runs/` y `logs/`.

## Cita del Dataset

```
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
```

## Licencia

Este proyecto está disponible bajo la licencia MIT.

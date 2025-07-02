# Detalles Técnicos del Entrenamiento sobre HAM10000

Este documento describe **de forma exhaustiva y técnica** el pipeline de preparación de datos, entrenamiento, optimización de hiperparámetros y evaluación del modelo Vision Transformer (ViT) desarrollado para la clasificación de lesiones cutáneas en el dataset **HAM10000**.

---

## 1. Dataset

| Concepto | Descripción |
|----------|-------------|
| Dataset  | HAM10000: 10 015 imágenes dermatoscópicas etiquetadas en 7 clases clínicas (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`). |
| Metadatos | Fichero `data/HAM10000_metadata.csv` con variables demográficas y de localización (`age`, `sex`, `localization`). |
| Imágenes  | Carpeta `data/HAM10000_images/`; nombres de archivo basados en `image_id` + `.jpg`. |

Los metadatos se sincronizan con las imágenes construyendo la ruta del archivo en tiempo de ejecución:
```python
# scripts/run_training.py (extracto)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg"))
```

---

## 2. Pre-procesamiento

1. **Validación de archivos** ‑ se descartan filas cuyo `image_path` no exista sobre disco.
2. **Estandarización de variables adicionales** (si está activado `data.use_additional_features`):
   * `age`: imputación con la media y posterior normalización Z-score mediante `StandardScaler`.
   * `sex` y `localization`: codificación entera vía `LabelEncoder`.
3. **Sampling opcional** ‑ controlado por `data.sample` y `data.sample_size` en `settings/config.yaml`.
4. **División** ‑ estratificada 80 % entrenamiento, 10 % validación, 10 % test (`train_test_split` con `stratify`).

---

## 3. Aumentación y Normalización de Imágenes

Definida en `classes/transforms.py` a partir del bloque `data.augmentation` del YAML:

```yaml
augmentation:
  use: true
  transformations:
    - name: "RandomHorizontalFlip"
      params:
        p: 0.5
    - name: "RandomRotation"
      params:
        degrees: 15
```

Pipeline resultante (orden conservado):
1. `RandomHorizontalFlip(p=0.5)`
2. `RandomRotation(degrees=15)`
3. `Resize((224, 224))` (bilineal)
4. `ToTensor()`
5. `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

---

## 4. Arquitectura del Modelo

Se utiliza el **Vision Transformer base** de `timm` (`vit_base_patch16_224`).

| Parámetro clave | Valor | Significado |
|-----------------|-------|------------|
| `patch_size`    | 16×16 | Tamaño de parches en los que se divide la imagen. |
| `embed_dim`     | 768   | Dimensión del embedding por parche. |
| `depth`         | 12    | Nº de bloques Transformer. |
| `num_heads`     | 12    | Cabezas de atención multi-cabeza. |
| `mlp_ratio`     | 4.0   | Escalado de la capa feed-forward. |
| `num_classes`   | 7     | Salida para las 7 clases de HAM10000. |
| `pretrained`    | True  | Pesos inicializados con ImageNet-21k para acelerar convergencia. |

### 4.1 Extensión para variables adicionales

Si `data.use_additional_features = True`, `models/vision_transformer.py` añade un *head* paralelo:
```python
model.additional_fc = nn.Sequential(
    nn.Linear(3, 32),  # age, sex, localization
    nn.ReLU(),
    nn.Linear(32, 16)
)
# Concatenación de [CLS] embedding (768) + 16 → FC final a 7 clases
```

---

## 5. Configuración de Entrenamiento (`settings/config.yaml`)

| Bloque | Clave | Valor | Explicación |
|--------|-------|-------|-------------|
| `training` | `epochs` | 20 | Nº de pasadas completas sobre el set de entrenamiento. |
|  | `learning_rate` | 1 × 10⁻⁵ | LR inicial para `AdamW`. |
|  | `scheduler_step_size` | 6 | Cada 6 épocas el LR es escalado. |
|  | `scheduler_gamma` | 0.7 | Factor multiplicativo del `StepLR`. |
| `data` | `batch_size` | 16 | Nº de muestras por batch inicial. Puede cambiar tras la búsqueda de hiperparámetros. |
| `logging` | `log_file` | logs/training.log | Ruta de logs con formato `logging`. |
| `tensorboard` | `log_dir` | runs/ham10000_experiment | Carpeta de eventos para TensorBoard. |

---

## 6. Búsqueda de Hiperparámetros

Implementada en `scripts/run_training.py -> optimize_hyperparameters` mediante **grid search exhaustivo** sobre las siguientes listas:

* `learning_rate_range`: `[1e-6, 1e-5, 1e-5]` (el valor repetido representa 5 × 10⁻⁶ en versiones previas).
* `batch_size_options`: `[8, 16]`
* `scheduler_gamma_range`: `[0.6, 0.7, 0.8]`

Para cada combinación se entrena el modelo durante `training.epochs` (20) y se guarda la última exactitud de validación. El mejor conjunto reportado en los comentarios del YAML es:

```
Best hyperparameters: {'learning_rate': 1e-05, 'batch_size': 16, 'scheduler_gamma': 0.7} with validation accuracy: 0.8886
```

Ese conjunto se reutiliza para el **entrenamiento final**, re-instanciando el modelo y los `DataLoader`.

---

## 7. Bucle de Entrenamiento (`fit/train.py`)

1. **Forward pass**
2. **Cálculo de pérdida**: `CrossEntropyLoss`.
3. **Backward**: `loss.backward()`.
4. **Actualización**: `optimizer.step()`.
5. **Scheduler**: `scheduler.step()` (al final de cada época).
6. **Criterio de mejor modelo**: mayor exactitud en validación (`epoch_acc`).
7. **Persistencia**: `torch.save(best_model_wts, 'models/best_model.pth')`.
8. **Logging + TensorBoard**: pérdidas y exactitudes por fase.
9. **Métricas agregadas** almacenadas en `metrics.json`.

---

## 8. Evaluación

Se ejecuta `fit/evaluate.py` sobre el loader de test:
* **Métricas**: `classification_report` de *scikit-learn* (precisión, recall, F1 por clase) y matriz de confusión.
* Salida persistida en JSON (`classification_report_test.json`) y CSV (`confusion_matrix_test.csv`).

Gráficas de progreso (`evaluate/*.png`) se generan offline a partir de `metrics.json`.

---

## 9. Reproducibilidad y Estructura de Proyecto

```
Tesis_HAM10000/
├── data/                      # Imágenes + metadatos
├── scripts/run_training.py    # Punto de entrada principal
├── settings/config.yaml       # Configuración declarativa
├── models/vision_transformer.py
├── fit/                       # Rutinas de train / eval
└── evaluate/                  # Post-procesado de métricas
```

* El **único** fichero que requiere modificación para experimentar es `settings/config.yaml`.
* Todos los artefactos (pesos, logs, eventos TensorBoard) se escriben en subcarpetas (`models/`, `runs/`, `logs/`).

---

## 10. Ejecución Paso a Paso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ajustar parámetros en settings/config.yaml (opcional)

# 3. Lanzar entrenamiento + búsqueda de hiperparámetros
python scripts/run_training.py  
#   ├── genera models/best_hyperparameter_model.pth
#   ├── guarda metrics.json, logs, tensorboard events
#   └── exporta classification_report_test.json & confusion_matrix_test.csv

# 4. Visualizar métricas
tensorboard --logdir runs/ham10000_experiment
```

> **Nota:** el script detecta automáticamente si hay GPU disponible (`torch.cuda.is_available()`), de lo contrario se ejecuta en CPU.

---

## 11. Versionado de Experimentos

Las ejecuciones de TensorBoard se almacenan en `runs/` con subcarpetas nombradas por experimento. Esto permite comparar distintas configuraciones sin sobrescribir resultados anteriores.

---

## 12. Referencias

* Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
* Touvron et al., *Training data-efficient image transformers & distillation through attention*. ICML 2021.
* Tim Salimans & Richard Chen, *Weight decay regularization* (AdamW). 

---

## 13. Glosario de Parámetros e Hiperparámetros

| Nombre | Tipo | Explicación | Implicación Práctica |
|--------|------|-------------|----------------------|
| `patch_size` | Arquitectura | Tamaño (en píxeles) de los parches extraídos de la imagen antes de ser proyectados como tokens. | Parches más pequeños capturan detalles finos pero incrementan la secuencia de entrada y el coste computacional. |
| `embed_dim` | Arquitectura | Dimensión del vector de embedding para cada parche/token. | Embeddings más grandes permiten representar más información pero requieren más parámetros y memoria. |
| `depth` | Arquitectura | Número de bloques Transformer apilados. | Mayor profundidad → capacidad expresiva más alta a costa de tiempo de entrenamiento y riesgo de sobreajuste. |
| `num_heads` | Arquitectura | Número de cabezas en la atención multi-cabeza. | Más cabezas permiten que el modelo atienda a múltiples sub-espacios, mejorando la capacidad de capturar relaciones globales. |
| `mlp_ratio` | Arquitectura | Factor de expansión de la capa feed-forward (FFN) interna. | Aumentar el ratio amplía la capacidad no lineal del modelo, pero encarece la computación. |
| `pretrained` | Arquitectura | Booleano que indica si se cargan pesos pre-entrenados en ImageNet-21k. | Inicializar con pesos pre-entrenados acelera la convergencia y mejora el rendimiento en datasets pequeños. |
| `use_additional_features` | Datos/Modelo | Activa la vía adicional para `age`, `sex`, `localization`. | Permite inyectar metadatos clínicos; puede mejorar la discriminación cuando la imagen es ambigua. |
| `learning_rate` | Entrenamiento | Tasa a la que se actualizan los pesos durante la optimización. | LR demasiado alto → divergencia; demasiado bajo → convergencia lenta o atrapada en mínimos locales. |
| `batch_size` | Entrenamiento | Nº de muestras procesadas antes de un paso de optimización. | Batches grandes estabilizan el gradiente pero requieren más memoria; tamaños pequeños introducen ruido útil para la generalización. |
| `epochs` | Entrenamiento | Pasadas completas por el set de entrenamiento. | Pocas épocas pueden sub-ajustar; demasiadas pueden sobre-ajustar si no hay regularización. |
| `optimizer` (`AdamW`) | Entrenamiento | Algoritmo que actualiza los pesos combinando momentum con adaptación por dimensión y decaimiento de peso. | Maneja bien curvas de error ruidosas; el término `weight_decay` regulariza al penalizar grandes pesos. |
| `scheduler_step_size` | Entrenamiento | Nº de épocas antes de aplicar decay al learning rate. | Controla cuán frecuentemente se reduce la LR; pasos cortos aceleran la reducción, potencialmente mejorando convergencia. |
| `scheduler_gamma` | Entrenamiento | Factor multiplicativo aplicado a la LR cuando el scheduler se activa. | Valores <1 reducen la LR; gamma muy bajo puede congelar el entrenamiento prematuramente. |
| `augmentation` | Pre-procesamiento | Conjunto y probabilidad de transformaciones aleatorias aplicadas a las imágenes. | Incrementa la diversidad del set de entrenamiento y reduce sobreajuste. |
| `StandardScaler` | Pre-procesamiento | Normaliza la variable `age` a media 0 y varianza 1. | Facilita el entrenamiento de la red adicional evitando que rangos distintos dominen el aprendizaje. |
| `LabelEncoder` | Pre-procesamiento | Convierte categorías (`sex`, `localization`, `dx`) en enteros. | Permite que los datos categóricos se integren en redes neuronales y métricas de evaluación. |
| `StepLR` | Scheduler | Reduce la LR de forma escalonada cada `step_size` épocas. | Permite grandes pasos al inicio y finos ajustes al final, mejorando la fine-tuning. |
| `CrossEntropyLoss` | Función de pérdida | Mide la discrepancia entre la distribución predicha y la verdadera. | Adecuada para clasificación multiclase; minimiza log-pérdida, equivalente a maximizar la probabilidad. | 

---

## 14. Curvas ROC Multiclase

La **ROC (Receiver Operating Characteristic)** es una curva que muestra la relación entre:

* **TPR (True Positive Rate)** \(= \frac{TP}{TP + FN}\) — también llamada *sensibilidad* o *recall*.
* **FPR (False Positive Rate)** \(= \frac{FP}{FP + TN}\).

Para cada umbral de decisión se obtiene un punto \((\text{FPR},\text{TPR})\); la curva completa refleja el compromiso entre sensibilidad y especificidad. El **AUC (Area Under the Curve)** cuantifica el área bajo la ROC (valor 1: separador perfecto, 0.5: azar).

### Multiclase (One-vs-Rest)
Para más de dos clases se aplica la estrategia **one-vs-rest**:
1. Se binariza la etiqueta verdadera para cada clase \(c\): positivo si la instancia pertenece a \(c\), negativo en caso contrario.
2. Se calcula la ROC y el AUC por clase.
3. Se añaden promedios:
   * **Micro-average**: calcula FPR/TPR considerando todas las instancias como un problema binario global.
   * **Macro-average**: promedia las ROC individuales interpolando sobre un grid común.

### Generación sin Reentrenar
El script `evaluate/roc_curve.py` automatiza el proceso:

```bash
python evaluate/roc_curve.py  # crea roc_curve_test.png
```

Pasos internos:
1. Carga el modelo entrenado (`scripts/models/final_model.pth` o `models/best_model.pth`).
2. Reconstruye el *test set* (10 % del dataset) con las mismas transformaciones de validación.
3. Propaga las imágenes, calcula probabilidades (`Softmax`).
4. Binariza etiquetas y calcula `fpr`, `tpr`, `auc` por clase más micro/macro.
5. Dibuja y guarda la figura `roc_curve_test.png` con todas las curvas y sus AUC.

Esta métrica complementa la precisión/F1 al mostrar el rendimiento a distintos umbrales de clasificación. 
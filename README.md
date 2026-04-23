# Clasificador de Sentimiento con Transformers

Proyecto académico de **Deep Learning** enfocado en la **clasificación de texto** mediante **Transformers**, utilizando **fine-tuning** de un modelo **DistilBERT multilingüe** para análisis de sentimientos sobre reseñas de películas. :contentReference[oaicite:1]{index=1}

## Objetivo

Desarrollar un modelo capaz de clasificar reseñas de texto en **sentimiento positivo o negativo**, aprovechando arquitecturas modernas basadas en Transformers para superar las limitaciones de enfoques clásicos como BoW, TF-IDF, Naive Bayes o SVM. :contentReference[oaicite:2]{index=2}

## Descripción del proyecto

En este trabajo se realizó el fine-tuning del modelo:

`lxyuan/distilbert-base-multilingual-cased-sentiments-student`

sobre el dataset **IMDb Movie Reviews**, con el objetivo de clasificar reseñas en dos clases:

- **0** → negativo
- **1** → positivo

Se evaluó el desempeño del modelo con métricas de clasificación y se implementó una demo funcional en **Hugging Face Spaces**. :contentReference[oaicite:3]{index=3}

## Dataset

Se utilizó el dataset **IMDb Movie Reviews**, que contiene:

- **50,000 reseñas**
- **25,000 para entrenamiento**
- **25,000 para prueba**
- distribución balanceada entre clases positivas y negativas :contentReference[oaicite:4]{index=4}

## Stack tecnológico

- **Python 3.12**
- **PyTorch 2.2**
- **Transformers 4.45**
- **Hugging Face Datasets**
- **Google Colab**
- **GPU Tesla T4 (16 GB VRAM)** :contentReference[oaicite:5]{index=5}

## Metodología

### 1. Carga y preprocesamiento de datos
- Se cargó el dataset IMDb usando la librería `datasets` de Hugging Face.
- No se aplicó resampling, ya que las clases estaban balanceadas.
- No se realizó limpieza agresiva del texto, para conservar la información original. :contentReference[oaicite:6]{index=6}

### 2. Tokenización
- Se utilizó el tokenizador del checkpoint base.
- Se tokenizó el campo de texto con truncamiento a **512 tokens**.
- El padding no se aplicó de forma fija en el preprocesamiento.
- Se usó `DataCollatorWithPadding` para padding dinámico por batch. :contentReference[oaicite:7]{index=7}

### 3. Modelo
Se utilizó un modelo de clasificación basado en **DistilBERT multilingüe**:

- `AutoModelForSequenceClassification`
- clasificación binaria: **positivo / negativo**
- ajuste de la capa de salida para **2 clases**
- uso de `ignore_mismatched_sizes=True` para reemplazar la capa original de clasificación :contentReference[oaicite:8]{index=8}

### 4. Evaluación
Las métricas utilizadas fueron:

- **Accuracy**
- **F1-score**

Se respetó el split original de IMDb (**train/test**) y se evaluó el modelo por época, conservando el mejor resultado según F1-score. También se usó una semilla fija (`seed=42`) para reproducibilidad. :contentReference[oaicite:9]{index=9}

## Resultados

Resultados del fine-tuning:

| Época | Train Loss | Val Loss | Accuracy | F1-score |
|------:|-----------:|---------:|---------:|---------:|
| 1     | 0.3095     | 0.2396   | 0.9052   | 0.9067   |
| 2     | 0.1878     | 0.2342   | 0.9143   | 0.9140   |
| 3     | 0.1308     | 0.3311   | 0.9138   | 0.9143   |

### Interpretación
- La pérdida de entrenamiento disminuyó de forma consistente.
- La pérdida de validación aumentó al final, sugiriendo **overfitting**.
- El modelo alcanzó un **F1-score final de 0.9143**, superando el objetivo del curso. :contentReference[oaicite:10]{index=10}

## Principales hallazgos

- DistilBERT multilingüe mostró un muy buen desempeño en análisis de sentimientos.
- El modelo logró generalizar adecuadamente pese a ser multilingüe.
- La capa de clasificación reajustada se adaptó bien al esquema binario del dataset.
- El entrenamiento pudo completarse en tiempos razonables usando recursos accesibles como Google Colab.
- La demo en Hugging Face Spaces añade valor práctico al proyecto. :contentReference[oaicite:11]{index=11}

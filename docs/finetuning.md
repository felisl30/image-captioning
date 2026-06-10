# Fine-tuning de BLIP para radiografías — guía conceptual

## Enfoque: transfer learning con descongelamiento parcial

BLIP ya sabe dos cosas que son valiosas: procesar imágenes (encoder ViT preentrenado sobre imágenes naturales) y generar texto descriptivo corto (decoder preentrenado sobre COCO). La pregunta es cuánto hay que tocar para que aprenda el dominio médico.

La respuesta es **transfer learning con descongelamiento parcial**, no fine-tuning completo. Actualizar todos los pesos con 10k muestras tiene riesgo real de degradar lo que el modelo ya sabe sin ganar suficiente conocimiento del dominio nuevo. El objetivo es quirúrgico: adaptar las capas que codifican semántica de alto nivel y lenguaje, preservar las que capturan features genéricos.

---

## Target del entrenamiento: `impression`, no `findings`

El texto de referencia es el campo `impression` del dataset — la conclusión clínica resumida (1–3 oraciones), no el campo `findings` (descripción detallada de hasta ~1.5k chars).

Esta decisión se tomó por cuatro razones:

1. **Compatibilidad con la arquitectura.** BLIP fue preentrenado para generar captions cortos (estilo COCO). Su decoder está optimizado para ese rango de longitud. Pedirle que genere 1.5k chars es pedirle algo distinto a lo que aprendió a hacer; con transfer learning parcial y 10k muestras, esa adaptación tiene alta probabilidad de fallar.

2. **Alineación con la evaluación.** Las métricas NLG (BLEU-4, CIDEr, METEOR) se calculan comparando el caption generado contra `impression` como referencia. Entrenar y evaluar sobre la misma distribución de texto es lo correcto.

3. **Claridad para el análisis de interpretabilidad.** Un caption de 5–15 palabras concentra la atención sobre pocas regiones específicas por palabra. Con un texto largo, la atención se diluye en decenas de tokens de relleno, haciendo el análisis de heatmaps más difícil.

4. **Menor riesgo con datos limitados.** `impression` es una distribución más cercana al estilo de COCO en longitud y estructura. La adaptación de dominio necesaria es más pequeña y alcanzable con 10k muestras.

`findings` podría ser interesante si se tuviera 5–10× más datos o si se entrenara el modelo desde cero. Con este presupuesto de cómputo y datos, `impression` es la elección correcta.

---

## Estrategia de descongelamiento

BLIP-base tiene un encoder ViT con 12 bloques transformer y un decoder de texto con 12 bloques. La misma lógica se aplica simétricamente a ambos: **congelar las capas que capturan conocimiento genérico, descongelar las que necesitan aprender el dominio médico.**

### Encoder ViT: congelar early, descongelar late

Los bloques 1–8 detectan features de bajo nivel — bordes, gradientes, texturas. Son útiles en cualquier dominio visual y costosos de re-aprender; tocarlos sin suficientes datos destruye representaciones que el modelo usa como base para todo lo demás.

Los bloques 9–12 codifican semántica de alto nivel. En el modelo base esa semántica es "perro", "auto", "árbol". Necesitamos que aprenda a reconocer estructuras anatómicas como consolidaciones, silueta cardíaca o seno costodiafragmático. Además, si el encoder entero se congela, los Grad-CAM antes y después del fine-tuning serían casi idénticos (mismas activaciones, mismos gradientes). Descongelar los últimos bloques del encoder es lo que hace que el análisis comparativo de interpretabilidad sea significativo.

### Decoder: la misma lógica aplica

El decoder ya sabe "hablar": genera texto coherente y gramaticalmente correcto. Los primeros bloques (1–4) codifican estructura lingüística general — sintaxis, coherencia, cómo encadenar ideas. Congelarlos preserva esa capacidad.

Los bloques tardíos (5–12) son más específicos de dominio: vocabulario, estilo, patrones de generación. Estos necesitan aprender el registro clínico de las impresiones radiológicas. El LM head (que proyecta al vocabulario) también se descongelna para que los nuevos términos médicos tengan mayor peso.

Un matiz: la **cross-attention está distribuida en todos los bloques del decoder**, no solo en los últimos. Al congelar los bloques 1–4, esas capas de cross-attention no se adaptan. Esto es aceptable — los bloques 5–12 con cross-attention descongelada tienen suficiente capacidad para aprender a "preguntar" cosas médicas al encoder visual.

### Tabla de configuración

| Componente | Estado | Learning rate |
|---|---|---|
| ViT patch embedding + positional embedding | Congelado | — |
| ViT bloques 1–8 | Congelado | — |
| ViT bloques 9–12 (últimos 4) | Descongelado | `5e-6` |
| Decoder bloques 1–4 (primeros 4) | Congelado | — |
| Decoder bloques 5–12 + LM head | Descongelado | `1e-5` |

El LR diferencial es importante: el encoder descongelado usa un LR más bajo para adaptarse suavemente, el decoder tardío puede aprender más agresivo porque su cambio de dominio es mayor.

Para implementar esto en PyTorch se pasan grupos de parámetros al optimizador:

```python
# Pseudo-código — implementación real en finetuner.py
encoder_late = model.vision_model.encoder.layers[8:]   # bloques 9-12
decoder_late = model.text_decoder.bert.encoder.layer[4:]  # bloques 5-12
lm_head = model.text_decoder.cls

optimizer = AdamW([
    {"params": encoder_late.parameters(), "lr": 5e-6},
    {"params": list(decoder_late.parameters()) + list(lm_head.parameters()), "lr": 1e-5},
], weight_decay=0.01)
```

---

## El objetivo de entrenamiento

BLIP se entrena con **language modeling loss**: dada la imagen y el texto de referencia (`impression`), el modelo aprende a predecir cada token a partir de los anteriores. Es una cross-entropy sobre el vocabulario en cada posición del texto.

Los tokens de padding se enmascaran con `-100` en los labels para que no contribuyan al loss — esto ya lo maneja `MimicCXRDataset`.

---

## Datos de entrada

Cada batch contiene:

- `pixel_values` — tensor de imagen normalizado, shape `(B, 3, 384, 384)` (el processor redimensiona automáticamente)
- `input_ids` — tokens del `impression` de referencia, shape `(B, L)`
- `attention_mask` — máscara de padding del texto, shape `(B, L)`
- `labels` — igual a `input_ids` pero con `-100` en los pads

El `MimicCXRDataset` ya recibe `text_col="impression"` por defecto, así que no requiere cambio en el código de datos.

---

## Hiperparámetros centrales

| Parámetro | Valor | Razón |
|---|---|---|
| LR encoder (bloques 9–12) | `5e-6` | Adaptación suave para no destruir features visuales |
| LR decoder | `1e-5` | Estándar para fine-tuning de decoders de transformers |
| Batch size efectivo | 8 | Equilibrio entre memoria VRAM y estabilidad del gradiente |
| Épocas | 3–5 | Con 24k muestras e `impression` corto, converge relativamente rápido |
| Optimizador | AdamW | Estándar para transformers; weight decay regulariza |
| Scheduler | Lineal con warmup | Los primeros pasos suben gradualmente el LR |
| Warmup steps | ~5% del total de pasos | Regla empírica para datasets medianos |
| Max length texto | 128 tokens | Suficiente para `impression` (1–3 oraciones); `findings` requeriría 512+ |
| Seed | 42 | Reproducibilidad del shuffling del DataLoader |

Si la GPU tiene menos de 16 GB de VRAM, usar **gradient accumulation**: con `accumulation_steps=4` y `batch_size=2` el batch efectivo sigue siendo 8.

---

## Estructura del loop de entrenamiento

1. **Fase train** — forward pass, cálculo del loss, backward, clip de gradientes (norma máxima 1.0), paso del optimizador y el scheduler. Se loguea el loss promedio de la época.

2. **Fase val** — sin gradientes. Se calcula el loss promedio sobre el DataLoader de validación. Este número es el criterio de selección del mejor checkpoint.

3. **Guardado de checkpoint** — al final de cada época se guarda en `models/blip_finetuned/epoch_N/`. Si el val loss es el mejor hasta ahora, se sobreescribe también `models/blip_finetuned/best/`.

4. **Early stopping** — si el val loss no mejora durante 2 épocas consecutivas se detiene el entrenamiento.

---

## Guardado y carga de checkpoints

`model.save_pretrained(ruta)` guarda los pesos y la configuración en una carpeta. Para cargarlo: `BlipForConditionalGeneration.from_pretrained(ruta)`. El processor se guarda y carga igual.

El checkpoint en `best/` es el que se usa en toda la comparación antes/después. **No modificar ni regenerar** una vez que empieza el análisis de interpretabilidad.

---

## Qué se monitorea durante el training

- **Train loss por época** — debería bajar. Si sube, el LR es demasiado alto.
- **Val loss por época** — el indicador real de generalización.
- **Gap train/val** — si divergen, hay sobreajuste. Con `impression` y 3 épocas es poco probable, pero posible si se descongelan demasiadas capas.

Las métricas NLG (BLEU, CIDEr, METEOR) no se calculan durante el training — son caras y se corren una sola vez al final en `05_resultados_y_metricas.ipynb` sobre el test set (~1.500 muestras).

---

## Dónde correr el fine-tuning

Con descongelamiento parcial (encoder bloques 9–12 + decoder bloques 5–12), BLIP-base + batch 8 ronda ~8–9 GB de VRAM — menos que el full fine-tuning porque no se computan gradientes sobre los parámetros congelados.

- **Kaggle Notebooks** — GPU T4 (16 GB). Gratuita, límite 30h semanales. Suficiente para 3 épocas sobre 10k muestras.
- **Google Cloud / Colab Pro** — GPU L4 (24 GB), más cómoda.

Workflow:
1. Subir `src/` y `data/splits/` a la plataforma cloud.
2. Ejecutar `notebooks/03_finetuning.ipynb`.
3. Descargar `models/blip_finetuned/best/` al entorno local.

---

## Relación con el análisis de interpretabilidad

El fine-tuning es un medio, no un fin. Lo importante es que después se pueda correr el mismo pipeline de cross-attention + Grad-CAM sobre las mismas 20–30 radiografías de `selected_indices.json` y comparar los heatmaps.

Descongelar los últimos bloques del encoder (no solo el decoder) es una decisión tomada en parte para que esta comparación sea significativa: si el encoder no cambia nada, los Grad-CAM antes y después del fine-tuning son prácticamente idénticos. Con los bloques 9–12 descongelados, ambos métodos de interpretabilidad tienen la oportunidad de mostrar cambios.
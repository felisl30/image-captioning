# Análisis post-VM — cómo trabajar con los outputs del notebook 07

Una vez que el notebook `07_explicabilidad_comparada_v0.ipynb` termina de correr
en la VM, todo el análisis posterior se hace en CPU local. Este documento explica
qué se generó, cómo cargarlo y cómo armar visualizaciones personalizadas.

---

## 1. Bajar los outputs de la VM

```bash
gcloud compute scp --recurse \
  vm-vision-final:~/image-captioning/outputs/notebook_comparativo \
  ./outputs/ \
  --zone=us-central1-a \
  --project=proyecto-final-im-captioning
```

Apagar la VM después de bajar los datos:

```bash
gcloud compute instances stop vm-vision-final \
  --zone=us-central1-a \
  --project=proyecto-final-im-captioning
```

---

## 2. Estructura de outputs generados

```
outputs/notebook_comparativo/
├── _images/                        ← 25 radiografías guardadas como PNG
│   ├── idx_0731.png
│   ├── idx_2296.png
│   └── ...
├── captions/
│   └── captions_bestof3.json       ← captions generadas por cada modelo
├── heatmaps/
│   └── idx_<NNN>/
│       ├── base/
│       │   ├── original.png
│       │   └── explanation.png     ← figura pre-renderizada (todos los tokens)
│       ├── ft5k/
│       └── ft10k/
├── arrays/                         ← LOS DATOS CRUDOS para análisis propio
│   ├── idx_0731__base__post_softmax.npz
│   ├── idx_0731__base__qk_logits.npz
│   ├── idx_0731__base__gradcam.npz
│   ├── idx_0731__ft5k__post_softmax.npz
│   └── ...  (25 imágenes × 3 modelos × 3 métodos = 225 archivos)
├── metrics/
│   ├── spatial_per_token.csv       ← métricas token a token
│   └── spatial_summary.csv        ← métricas agregadas por (idx, modelo, par)
└── summary.csv                     ← un renglón por (idx, modelo)
```

---

## 3. Qué contiene cada archivo `.npz`

Cada `.npz` en `arrays/` corresponde a una combinación `(imagen, modelo, método)`
y tiene tres keys:

| Key | Tipo | Contenido |
|---|---|---|
| `words` | `array` de strings | Tokens filtrados (sin stopwords ni puntuación) |
| `heatmaps` | `float32 (n_tokens, 24, 24)` | Mapa de atención por token sobre la grilla 24×24 |
| `caption` | `array` de string | La caption completa generada |

Los tokens en `words` ya pasaron por `token_filter.py` — se descartaron stopwords,
puntuación y tokens especiales. Lo que queda son tokens con contenido semántico,
que pueden ser médicos o no.

```python
import numpy as np

z = np.load("outputs/notebook_comparativo/arrays/idx_0731__ft10k__post_softmax.npz",
            allow_pickle=True)

words   = z["words"]      # ej: ["bilateral", "pleural", "effusion", "lung"]
heatmap = z["heatmaps"]   # shape: (4, 24, 24)
caption = str(z["caption"])
```

---

## 4. Visualización personalizada — solo tokens médicos

`token_filter.py` define el vocabulario médico en `MEDICAL`. Podés usarlo para
filtrar los heatmaps y mostrar solo los tokens clínicamente relevantes:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.interpretability.token_filter import MEDICAL, normalize_token

def load_medical_heatmaps(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    words = z["words"]
    heatmaps = z["heatmaps"]

    mask = [normalize_token(w) in MEDICAL for w in words]
    return words[mask], heatmaps[mask]


def plot_medical_heatmaps(image_path, npz_path, alpha=0.5, title=""):
    image = Image.open(image_path).convert("RGB").resize((384, 384))
    words, heatmaps = load_medical_heatmaps(npz_path)

    if len(words) == 0:
        print("No hay tokens médicos en este ejemplo.")
        return

    n = len(words)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, (word, hmap) in enumerate(zip(words, heatmaps)):
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_resized = np.array(
            Image.fromarray((hmap_norm * 255).astype("uint8")).resize((384, 384))
        ) / 255.0

        axes[i + 1].imshow(image)
        axes[i + 1].imshow(hmap_resized, alpha=alpha, cmap="jet")
        axes[i + 1].set_title(word, fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Uso:
plot_medical_heatmaps(
    image_path="outputs/notebook_comparativo/_images/idx_0731.png",
    npz_path="outputs/notebook_comparativo/arrays/idx_0731__ft10k__post_softmax.npz",
    title="ft10k / post_softmax — tokens médicos",
)
```

---

## 5. Comparar base vs finetuned para el mismo token

Esto no está implementado en el pipeline actual pero se puede hacer directamente
con los arrays. La idea es: para el mismo token médico en la misma imagen,
comparar el heatmap de `base` vs `ft5k` vs `ft10k`:

```python
def compare_models_for_token(image_path, idx_str, token, models=("base", "ft5k", "ft10k"),
                              method="post_softmax"):
    image = Image.open(image_path).convert("RGB").resize((384, 384))
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3 * (len(models) + 1), 3))

    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, model_tag in enumerate(models):
        path = f"outputs/notebook_comparativo/arrays/{idx_str}__{model_tag}__{method}.npz"
        z = np.load(path, allow_pickle=True)
        words = list(z["words"])
        heatmaps = z["heatmaps"]

        token_norm = normalize_token(token)
        matches = [j for j, w in enumerate(words) if normalize_token(w) == token_norm]

        ax = axes[i + 1]
        if not matches:
            ax.set_title(f"{model_tag}\n(sin '{token}')")
            ax.axis("off")
            continue

        hmap = heatmaps[matches[0]]
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_resized = np.array(
            Image.fromarray((hmap_norm * 255).astype("uint8")).resize((384, 384))
        ) / 255.0

        ax.imshow(image)
        ax.imshow(hmap_resized, alpha=0.5, cmap="jet")
        ax.set_title(f"{model_tag}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"'{token}' — {method} — {idx_str}")
    plt.tight_layout()
    plt.show()


# Uso:
compare_models_for_token(
    image_path="outputs/notebook_comparativo/_images/idx_0731.png",
    idx_str="idx_0731",
    token="effusion",
    method="post_softmax",
)
```

---

## 6. Leer las métricas agregadas

```python
import pandas as pd

summary = pd.read_csv("outputs/notebook_comparativo/summary.csv")
per_token = pd.read_csv("outputs/notebook_comparativo/metrics/spatial_per_token.csv")
spatial = pd.read_csv("outputs/notebook_comparativo/metrics/spatial_summary.csv")

# Ver métricas promedio por modelo y par de métodos
spatial.groupby(["model_tag", "method_a", "method_b"])[
    ["pearson_mean", "cosine_mean", "top10_iou_mean"]
].mean()
```

---

## 7. Qué análisis quedan por hacer

Con los arrays descargados, el trabajo restante en CPU local es:

1. **Visualización selectiva** — mostrar solo tokens médicos (sección 4).
2. **Comparación base vs finetuned** — para el mismo token, ¿el heatmap cambia?
   ¿Cambia hacia zonas más relevantes? (sección 5).
3. **Análisis de métricas** — ¿qué par de métodos coincide más? ¿Cambia la
   concordancia entre métodos después del finetuning? (sección 6).
4. **Selección de ejemplos para el paper** — elegir las imágenes y tokens que
   mejor ilustran los tres resultados posibles (A, B o C del CLAUDE.md).
5. **Figuras finales** — armar las figuras con DPI alto para el póster A0
   (`dpi=200` mínimo).

Nada de esto requiere GPU ni la VM encendida.

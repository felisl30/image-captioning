# Pendientes — análisis para presentación

Estas tres cosas no están en el plan actual (`docs/plan_analisis_presentacion.md`)
y deberían agregarse antes de armar las figuras finales.

---

## 1. Script de exploración de tokens disponibles

Antes de elegir qué imágenes mostrar en las figuras, necesitamos saber qué tokens
médicos tiene cada imagen en cada modelo. Sin esto hay que abrir npz a mano.

Lo que necesitamos:
- Para las 25 imágenes × 3 modelos: listar los tokens médicos disponibles
- Identificar qué imágenes tienen tokens como "effusion", "edema", "atelectasis"
  en ft10k pero NO en base (esos son los mejores casos para la figura central)
- Identificar qué imagen tiene el mayor número de tokens médicos distintos
  (mejor candidata para la figura de análisis por token)

Fuente: `outputs/notebook_comparativo/arrays/idx_<NNN>/`
Usar `MEDICAL` de `src/interpretability/token_filter.py` para filtrar.

---

## 2. Tabla referencia vs generada

En `outputs/notebook_comparativo/summary.csv` hay una columna `reference` con
la impresión clínica real de MIMIC-CXR. Incluir en la presentación la comparación:

```
referencia (ground truth) | base | ft5k | ft10k
```

Es la forma más directa de mostrar el salto de calidad del fine-tuning y ancla
el análisis en los hallazgos clínicos reales. Actualmente el plan no lo menciona.

Fuente: `outputs/notebook_comparativo/summary.csv` (columnas `reference`, `caption`, `model_tag`)
También: `outputs/notebook_comparativo/captions/captions_bestof3.json`

---

## 3. Figura del mode collapse — cascada de entropía

`analisis/01_mode_collapse_s1_d2.md` ya tiene todos los datos de la cascada:

```
step=0  "no"    p=0.21  entropy=4.38
step=1  "acute" p=0.56  entropy=2.21
step=2  "card"  p=0.84  entropy=0.85
step=3  "##io"  p=0.995 entropy=0.04
```

Es una figura muy visual para explicar por qué colapsa el modelo y distingue el TP
de un trabajo que solo reporta "greedy no funciona". Muestra que el modelo sí
aprendió distribuciones diversas (step 0 es competitivo) pero greedy amplifica
el ganador marginal hasta hacerlo determinista.

Datos en: `outputs/mode_collapse_debug/`
Referencia metodológica: `analisis/01_mode_collapse_s1_d2.md`

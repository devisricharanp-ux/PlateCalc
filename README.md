# PlateCalc🍽️ 
This Repository has all the resources to build a Computer vision based model( using YOLOv8) to predict the calories present in a plate

Dataset used - [foodseg103](https://www.kaggle.com/datasets/ggrill/foodseg103)
## 1.Image Preprocessing
   Use the annotated images from the Dataset to generate labels in YOLOtxt format.
   
   Use this boilerplate code [annotated to YOLO](https://github.com/devisricharanp-ux/PlateCalc/blob/code/annotated_to_YOLOtxt.py)
## 2. Dataset Preparation & Model Training
   Use the FoodSeg103 dataset containing food images, segmentation masks, and labels.
   
   I have used two methods to train the model

## Approach Comparison: Single-Stage vs Two-Stage Hierarchical

### Two-Stage Hierarchical (12-class detector + per-category classifiers)

**Pros**
- Each stage-2 classifier handles only 3–20 classes, easier to train with limited data per class
- Isolated failure — a weak classifier only affects its own category, not the whole system
- Easier debugging — separate stage-1 (detection) vs stage-2 (classification) issues
- Independent iteration — retrain one classifier without touching the rest
- Stage-1 (12-class) converges faster and more reliably than a 103-class detector

**Cons**
- Error propagation — wrong macro-category routing means stage 2 can never recover
- More moving parts — 1 detector + 12 classifiers = higher complexity and inference latency
- Crop quality dependency — stage 2 accuracy tied to stage-1 bounding box quality
- Imbalanced classifiers — category sizes range from 4 to 20 classes, uneven performance expected
- More complex deployment and maintenance pipeline

---

### Single-Stage (103-class YOLOv8 detector)

**Pros**
- Simple pipeline — one model, one inference pass, lower latency
- No error propagation — every class is considered for every detection
- Easier to deploy — single model file, single training run
- Joint learning of spatial + class features, can leverage contextual cues

**Cons**
- 103-class detection head is hard to train well with limited per-class data
- Visually similar classes (fruits, leafy greens) prone to confusion
- Class imbalance across 103 categories harder to manage
- Harder to debug — unclear if low accuracy is data, capacity, or architecture related
- Inflexible — improving one category requires retraining the entire model

---

### Recommendation
- Hierarchical approach is preferred given single-stage underperformance
- Mitigate routing errors via:
  - Monitoring stage-1 confusion matrix closely
  - Top-k routing (try top-2 macro-categories, pick higher-confidence final result)
- Benchmark both pipelines end-to-end on final 103-class accuracy and calorie estimation error
  ## Experimental Results

### Single Model — 103 Classes (YOLOv8s)

| Metric | precision(B) | recall(B) | mAP50(B) | mAP50-95(B) | precision(M) | recall(M) | mAP50(M) | mAP50-95(M) |
|---|---|---|---|---|---|---|---|---|
| Average | 43.68 | 31.29 | 28.89 | 23.45 | 44.19 | 31.34 | 29.11 | 22.43 |
| Maximum | 48.26 | 34.65 | 31.23 | 25.51 | 48.55 | 34.90 | 31.48 | 24.30 |

### Single Model — 103 Classes (YOLOv8n)

| Metric | precision(B) | recall(B) | mAP50(B) | mAP50-95(B) | precision(M) | recall(M) | mAP50(M) | mAP50-95(M) |
|---|---|---|---|---|---|---|---|---|
| Average | 45.28 | 24.90 | 22.40 | 17.87 | 45.87 | 24.73 | 22.40 | 17.17 |
| Maximum | 50.53 | 29.98 | 26.43 | 21.18 | 50.41 | 30.02 | 26.44 | 20.33 |

### Stage 1 — Macro Categories (12 Classes, YOLOv8s)

| Metric | precision(B) | recall(B) | mAP50(B) | mAP50-95(B) | precision(M) | recall(M) | mAP50(M) | mAP50-95(M) |
|---|---|---|---|---|---|---|---|---|
| Average | 47.55 | 41.47 | 40.84 | 31.79 | 48.28 | 41.10 | 40.74 | 30.34 |
| Maximum | 57.51 | 46.58 | 47.09 | 37.36 | 57.76 | 46.06 | 46.98 | 35.68 |

---

### Observations

- **Macro-category (12-class) model substantially outperforms both 103-class models** across every metric — mAP50 improves from ~29 (103-class YOLOv8s) to ~41 (12-class), and mAP50-95 nearly doubles from ~17–23 to ~30–32 on average
- **Recall sees the largest jump** (~25–31 → ~41–47), indicating the 103-class models were missing a large fraction of objects entirely — likely due to insufficient per-class samples for fine-grained categories
- **YOLOv8n (nano) on 103 classes underperforms YOLOv8s on 103 classes** in recall and mAP, despite having higher precision — suggests the smaller model struggles even more with the high class-count, low-data-per-class problem
- **Confirms the hierarchical hypothesis** — reducing class count from 103 → 12 meaningfully eases the detection task, supporting the two-stage pipeline approach over a single flat 103-class detector
- **Next step**: stage 2 classifier accuracy (per macro-category) will determine whether these stage-1 gains translate into better final fine-grained results — overall system accuracy = stage-1 mAP × stage-2 classifier accuracy (roughly), so both stages need to be evaluated together for the final comparison

  

# Elbatoul-NLP-W1-instruction-embeddings

Encodes 125 natural robot instructions across 5 command types (pick, push, move, put, find) using `all-MiniLM-L6-v2` and validates embedding quality through similarity metrics, visualization, and a classifier.

---

## How to run

Open `Elbatoul-NLP-W1-embeddings.ipynb` in Google Colab and run all cells top to bottom.

```bash
# All dependencies are installed in Cell 1
pip install sentence-transformers scikit-learn matplotlib seaborn pandas numpy
```

---

## Inputs / Outputs

**Input:** 125 hand-written natural language instructions (defined directly in Cell 3, no external file needed)

**Outputs:**
| File | Description |
|---|---|
| `nlp_instructions_125.csv` | Dataset of instructions + labels |
| `nlp_instructions_125_with_embeddings.pkl` | Dataset with 384-dim embedding vectors |
| `embeddings_125.npy` | Raw embedding matrix for RL team |
| `tsne_elbatoul_v2.png` | t-SNE cluster visualization |
| `pca_elbatoul_v2.png` | PCA visualization |
| `combined_visualization.png` | t-SNE + PCA side by side |

---

## Results

| Metric | Value | Target |
|---|---|---|
| Classifier accuracy | 92.0% | > 90% |
| Intra/Inter ratio | 1.31 | > 1.30 |
| Inter-class similarity | 0.37 | < 0.45 |
| Paraphrase pairs | 8/12 strict, 12/12 soft | > 0.85 |
| Cross-class pairs | 4/4 | < 0.50 |
| Encoding speed (cached) | < 1ms | < 10ms |

> Note: silhouette score (0.04) is low due to structural similarity between pick/push/put classes in `all-MiniLM-L6-v2`. Intra/inter ratio and classifier accuracy confirm meaningful separation exists in the full 384-dim space.

---

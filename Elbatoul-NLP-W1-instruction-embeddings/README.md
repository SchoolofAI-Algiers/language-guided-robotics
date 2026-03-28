# Elbatoul-NLP-W1-instruction-embeddings

Encodes 100 natural language robot instructions using sentence-transformers and validates embedding quality via cosine similarity, t-SNE, PCA, silhouette score, and classifier accuracy.

## How to run
```bash
pip install sentence-transformers scikit-learn matplotlib seaborn pandas numpy
jupyter notebook notebook.ipynb
```

## Inputs / Outputs

- Input: hardcoded instruction dataset (100 sentences, 5 action types)
- Output:
  - `instructions_elbatoul.csv` — instruction + action type
  - `embeddings_elbatoul.npy` — (100, 384) embedding matrix
  - `tsne_elbatoul.png` — t-SNE visualization
  - `pca_elbatoul.png` — PCA visualization

## Status

[in progress]
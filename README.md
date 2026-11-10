# HTF: A Token-Free Language Model with Morphological CNN for Brazilian Portuguese

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](HTF_article.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

Official repository for **"HTF: A Token-Free Language Model with Morphological CNN for Brazilian Portuguese"**  
*João L. Ortiz\*, Ivan R. Guilherme\*, Jonas Queiroz†*  
*(\*Universidade Estadual Paulista — UNESP, †Universidade Estadual de Campinas — UNICAMP)*

---

## 💡 Overview

Language models based on subword tokenization (BPE, WordPiece) suffer from fixed vocabularies, sensitivity to textual noise, and excessive fragmentation in morphologically rich languages like Portuguese.

**HTF (Hierarchical Token-Free)** is a **144M-parameter architecture** that operates directly on Unicode characters, imposing an explicit hierarchy on sub-lexical processing via a **multi-scale morphological CNN with intra-word attention**.

```
Input Text: "energia"
     │
     ▼
[Char Embedding] (64-dim)
     │
     ├── Conv1D (k=2) ──> Bigrams   (1024-dim)
     ├── Conv1D (k=3) ──> Trigrams  (1024-dim)
     ├── Conv1D (k=4) ──> 4-grams   (1024-dim)
     └── Conv1D (k=5) ──> 5-grams   (1024-dim)
     │
     ▼
[Concatenate & Intra-Word Multi-Head Self-Attention]
     │
     ▼
[Max-Pooling] ──> Dense Word Vector (1024-dim)
     │
     ▼
[Encoder & Decoder Transformers] ──> [Two-Level Autoregressive WordDecoder]
```

---

## 🚀 Key Results & Robustness

- **7× to 19× More Robust to Typos:** In controlled experiments with 10% character corruption (typos), HTF exhibits a cross-entropy degradation of only **+105%**, compared to **+692% for Pythia-160M** and **+1980% for GPT-2 PT**.
- **Cloze Test:** Retains **93.8% of correct token probability under 10% noise** (versus 71.1% for Pythia and 43.7% for GPT-2).
- **Format Generalization on RTE:** Responded in recognizable structured format in **99% of examples** (273/277), compared to 8.7% (24/277) for Tucano-160M.
- **Pre-trained Corpora:** Pre-trained on Carolina, Wikipedia PT, FineWeb-2, and refined on GigaVerbo (200B tokens).

---

## 📄 Citation

```bibtex
@article{ortiz2026htf,
  title={HTF: A Token-Free Language Model with Morphological CNN for Brazilian Portuguese},
  author={Ortiz, Jo{\~a}o L. and Guilherme, Ivan R. and Queiroz, Jonas},
  journal={Preprint},
  year={2026}
}
```

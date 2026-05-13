# 🧠 Sentiment & Fake News Detection

A multi-phase NLP project that tackles two distinct text classification problems:
**Sentiment Analysis** on tweets and **Fake News Detection** on political statements —
using a full progression from classical machine learning to deep learning and transformers.

---

## 📌 Project Overview

This project is split into two parallel tracks, each going through the same 3-phase pipeline:

| Track | Dataset | Task |
|---|---|---|
| **Sentiment Analysis** | Sentiment140 (1.6M tweets) | Classify tweets as Positive / Negative |
| **Fake News Detection** | LIAR (~12,500 political statements) | Classify statements as Fake / True |

Each track is built progressively — starting with simple baselines and working up to state-of-the-art transformer models — so the results at each phase tell a clear story about why more advanced models were developed.

---

## 🔄 Pipeline — 3 Phases

### Phase 1 — Classical Machine Learning

> Establishes baselines using traditional NLP + ML methods.

**Preprocessing:**
- Lowercasing, removing URLs, mentions, special characters
- TF-IDF vectorization (up to 8,000–20,000 features)
- Dimensionality reduction via TruncatedSVD (LSA)

**Models trained:**
- K-Nearest Neighbors (KNN) with GridSearchCV tuning
- Naive Bayes (Multinomial NB)
- Random Forest with GridSearchCV tuning
- XGBoost *(Fake News track only)*

**Evaluation:** Accuracy, F1, AUC-ROC, ROC curves, Confusion Matrix

---

### Phase 2 — Deep Learning

> Moves beyond bag-of-words to neural sequence models and pretrained embeddings.

**Multi-kernel CNN:**
- Embedding layer → parallel Conv1D kernels (sizes 2, 3, 4) → GlobalMaxPooling → Dense
- Training curves (loss & accuracy)

**Transfer Learning (5 Experiments):**

| Experiment | Embedding | Mode |
|---|---|---|
| 1 | Random init | Trainable |
| 2 | GloVe-Twitter-25 | Frozen |
| 3 | GloVe-Twitter-25 | Fine-tuned |
| 4 | FastText-300 | Frozen |
| 5 | FastText-300 | Fine-tuned |

Comparison via bar charts, heatmaps, and confusion matrix.

**Optimizer Comparison:**
- Adam vs SGD with Momentum on identical CNN architecture
- Tracks: loss, accuracy, AUC, training time, convergence speed
- Full comparison table + 5 visualizations

**Autoencoder *(Fake News track only)*:**
- LSTM-based autoencoder trained on real news only
- Detects fake news via reconstruction error thresholding

---

### Phase 3 — Recurrent Networks & Transformers

> Captures long-range sequential context — critical for nuanced political statements.

| Model | Key Idea |
|---|---|
| **Simple RNN** | Vanilla recurrence — baseline for Phase 3 |
| **GRU** | Adds Reset + Update gates to fix vanishing gradient |
| **LSTM** | Adds Forget + Input + Output gates for best memory control |
| **BERT (fine-tuned)** | Pretrained transformer fine-tuned on each dataset |

All four models are applied to **both tracks** and compared on Accuracy, F1, and AUC-ROC.

> **Note:** On the LIAR dataset, even state-of-the-art models achieve ~65–70% accuracy. Low accuracy on fake news is expected — LIAR is one of the hardest NLP benchmark datasets due to its politically nuanced statements that require real-world fact-checking to judge.

---

## 📊 Datasets

### Sentiment140
- **Source:** [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1,600,000 tweets
- **Labels:** 0 = Negative, 4 (→ 1) = Positive
- **Loaded via:** `kagglehub`

### LIAR
- **Source:** [Kaggle — LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)
- **Size:** ~12,500 political statements (pre-split into train / valid / test)
- **Original labels:** 6 classes (pants-fire, false, barely-true, half-true, mostly-true, true)
- **Binarized as:** `{false, barely-true, pants-fire}` → Fake (0), rest → True (1)
- **Loaded via:** `kagglehub`

---

## ⚙️ How to Run

### Option 1 — Google Colab (Recommended)

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells top to bottom
4. Kaggle datasets are downloaded automatically via `kagglehub` — no manual upload needed

### Option 2 — Jupyter Notebook (Local)

1. Clone the repo:
```bash
git clone https://github.com/your-username/Sentiment-Fake-News-Detection.git
cd Sentiment-Fake-News-Detection
```

2. Install dependencies:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn wordcloud kagglehub gensim xgboost transformers
```

3. Launch Jupyter and run the notebook:
```bash
jupyter notebook
```

> ⚠️ Running locally without a GPU will be significantly slower, especially for Phase 3 models.


---

## 📈 Results

All results (accuracy, F1, AUC-ROC, training curves, confusion matrices, and comparison tables) are available inside the notebook under each phase and experiment section.

---

## 💡 Key Takeaways

- **Classical ML** sets a fast, interpretable baseline but misses sequential context
- **CNN** captures local n-gram patterns well and trains fast
- **Transfer Learning** (GloVe / FastText) improves performance — fine-tuned embeddings consistently outperform frozen ones
- **RNN → GRU → LSTM** shows a clear accuracy progression, especially on the longer LIAR statements where long-range context matters most
- **BERT** achieves the best results on both tracks by leveraging deep bidirectional pretraining
- The LIAR dataset is fundamentally harder than Sentiment140 — model rankings are consistent across both tracks but absolute scores are lower for fake news

---

## 📄 License

This project is for academic purposes.

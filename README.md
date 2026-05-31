# TP4 — Sentiment Classification on IMDB (RNN / CNN / GloVe)

A practical deep learning lab exploring text sentiment classification using the IMDB movie review dataset. The notebook compares multiple architectures (Dense, CNN, LSTM, GRU, CNN+RNN) across multiple word embedding strategies (GloVe, Word2Vec, FastText, TF-IDF).

---

## Objective

Classify movie reviews as **positive** or **negative** using deep learning models fed by pre-trained and locally trained word embeddings.

---

## Dataset

**IMDB Dataset** — 50,000 labeled movie reviews (balanced: 25k positive / 25k negative).

| Split | Size |
|-------|------|
| Train | 80% |
| Test | 20% |

---

## Pipeline Overview

```
Raw reviews (.csv)
  → Text Preprocessing
  → Label Encoding (positive=1, negative=0)
  → Train/Test Split (80/20)
  → Tokenization + Padding (maxlen=100)
  → Embedding Matrix (GloVe / Word2Vec / FastText / TF-IDF)
  → Model Training
  → Evaluation & Comparison
```

---

## Preprocessing

Each review goes through these steps:
1. **HTML tag removal** — strips `<br />` and similar tags
2. **Non-alphabetic character removal** — removes punctuation and digits
3. **Isolated character removal** — cleans single-letter noise
4. **Whitespace normalization**

---

## Tokenization & Padding

- `Tokenizer(num_words=5000)` — builds a word-to-index vocabulary from the training set
- `texts_to_sequences` — converts each review to a list of integers
- `pad_sequences(maxlen=100)` — all reviews truncated or zero-padded to exactly 100 tokens

---

## Word Embeddings

### GloVe (pre-trained)
Loaded from `glove.6B.100d.txt` — 100-dimensional vectors pre-trained on Wikipedia. An embedding matrix of shape `(vocab_size, 100)` is built by looking up each tokenizer word in the GloVe dictionary. **Weights are frozen** (`trainable=False`) during training.

### Word2Vec (trained on IMDB)
Trained locally on the IMDB training corpus using Gensim (`vector_size=100, window=5, min_count=2, epochs=5`). Defines each word by the words surrounding it.

### FastText (trained on IMDB)
Same setup as Word2Vec but decomposes words into character n-grams — handles **out-of-vocabulary words** gracefully by approximating their vector from sub-word pieces.

### TF-IDF (baseline)
`TfidfVectorizer(max_features=20000)` + `LogisticRegression` — non-sequential bag-of-words representation used as a classical ML baseline.

---

## Models

### A. Dense Network
```
Embedding (frozen GloVe) → Flatten → Dense(1, sigmoid)
```
Loses word order entirely — "not good" and "good not" produce identical vectors.

### B. CNN (1D Convolutional)
```
Embedding → Conv1D(128, kernel=5, relu) → MaxPooling1D(2) → Flatten → Dense(1, sigmoid)
```
Captures **local word patterns** (n-gram-like features). Detects phrases like "very bad" or "not good".

### C. LSTM
```
Embedding → LSTM(128) → Dense(1, sigmoid)
```
Reads the review **word by word**, maintaining a hidden state that captures long-range dependencies.

### D. GRU
```
Embedding → GRU(128) → Dense(1, sigmoid)
```
Simplified version of LSTM — fewer parameters, faster training, comparable performance.

### E. CNN + RNN (Hybrid)
```
Embedding → Conv1D(128, 5, relu) → MaxPooling1D(2) → LSTM(128) → Dense(1, sigmoid)
```
CNN extracts local features first, then LSTM models the sequence of those features.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | Binary Cross-Entropy |
| Optimizer | Adam |
| Epochs | 6 |
| Batch size | 128 |
| Validation split | 20% of training set |

---

## Results Summary

| Model | GloVe | Word2Vec | FastText | TF-IDF (LogReg) |
|-------|-------|----------|----------|-----------------|
| LSTM | ~0.856 | ~0.860 | ~0.849 | — |
| GRU | ~0.859 | **~0.874** | ~0.863 | — |
| CNN+RNN | ~0.85x | ~0.85x | ~0.85x | — |
| LogReg | — | — | — | **~0.900** |

> Exact values depend on your run — fill in the table from your notebook outputs.

### Key Takeaways
- **TF-IDF + Logistic Regression** achieves the highest accuracy (~0.90), showing that a well-calibrated bag-of-words baseline can outperform deep sequential models on IMDB with only 6 training epochs.
- **GRU** is the best sequential model overall — good balance between capacity and regularization.
- **Word2Vec** outperforms GloVe and FastText for locally-trained embeddings.
- **Dense** is the weakest architecture — it discards word order completely.
- **CNN+RNN** underperforms relative to its complexity, likely due to optimization difficulty with only 6 epochs.

---

## Project Structure

```
TP4_DL_IMDB.ipynb          # Main notebook
IMDB Dataset/
  IMDB Dataset.csv          # Raw dataset
glove.6B.100d/
  glove.6B.100d.txt         # Pre-trained GloVe vectors (100d)
```

---

## Requirements

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn gensim
```

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building and training |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Plotting curves and distributions |
| Scikit-learn | Train/test split, TF-IDF, Logistic Regression |
| Gensim | Word2Vec and FastText training |

---

## Glossary

| Term | Definition |
|------|-----------|
| **GloVe** | Pre-trained vectors encoding global word co-occurrence statistics from Wikipedia |
| **Word2Vec** | Vectors trained to predict surrounding words (context defines meaning) |
| **FastText** | Like Word2Vec but decomposes words into character n-grams — handles unknown words |
| **TF-IDF** | Term Frequency × Inverse Document Frequency — weights word importance per document |
| **Padding** | Extending or truncating sequences to a fixed length (zeros appended at end) |
| **LSTM** | Long Short-Term Memory — RNN variant with gates to capture long-range dependencies |
| **GRU** | Gated Recurrent Unit — lighter LSTM with fewer parameters |
| **Binary Cross-Entropy** | Loss function for binary classification (positive / negative) |

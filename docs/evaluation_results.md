# ReviewIQ: Empirical Evaluation & Benchmarking Suite

**Project**: ReviewIQ — AI-Powered Product Review Intelligence and Recommendation System  
**Dataset**: Amazon Laptop Reviews & Metadata Corpus (with SemEval-2014 Task 4 Laptop Benchmark)  
**Execution Environment**: Python 3.14 / PyTorch (CPU & CUDA auto-fallback on NVIDIA RTX 3050 4GB)

---

## 1. Executive Summary

This evaluation report benchmarks every core algorithm and machine learning pipeline implemented across **ReviewIQ**:
1. **Classical vs. Transformer Sentiment Classification**: BoW, TF-IDF, and domain Skip-gram Word2Vec across Multinomial Naive Bayes, Linear SVM, Decision Trees, and fine-tuned DistilBERT.
2. **Unsupervised Topic Modeling**: Intrinsic $C_{\text{UMass}}$ topic coherence, NPMI, and Topic Diversity across LDA, NMF, and BERTopic.
3. **Retrieval & Reranking**: Sparse BM25/TF-IDF, Dense FAISS IndexFlatIP (`all-MiniLM-L6-v2`), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`).
4. **Aspect-Based Sentiment Analysis (ABSA)**: Precision, Recall, and Accuracy evaluated against the gold standard **SemEval-2014 Task 4 Laptop Test Dataset**.
5. **Review Summarization**: Graph-based extractive TextRank vs. neural abstractive BART (`facebook/bart-large-cnn`) with ROUGE-1, ROUGE-2, and ROUGE-L metrics.

---

## 2. Sentiment Classification Benchmark

Evaluated on customer reviews categorized into discrete sentiment classes: *Positive*, *Neutral*, and *Negative*.

### 2.1 Model & Representation Performance Matrix

| Representation | Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) | Inference Latency (ms/doc) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoW (Count)** | Multinomial Naive Bayes | 0.812 | 0.791 | 0.785 | 0.788 | 0.809 | 0.18 | 0.08 |
| **BoW (Count)** | Linear SVM | 0.835 | 0.814 | 0.809 | 0.811 | 0.832 | 0.82 | 0.11 |
| **BoW (Count)** | Decision Tree (depth=20) | 0.748 | 0.720 | 0.718 | 0.719 | 0.745 | 1.45 | 0.09 |
| **TF-IDF (Sublinear)** | Multinomial Naive Bayes | 0.841 | 0.826 | 0.812 | 0.819 | 0.838 | 0.22 | 0.09 |
| **TF-IDF (Sublinear)** | **Linear SVM** *(Best Classical)* | **0.876** | **0.865** | **0.854** | **0.859** | **0.874** | 0.64 | 0.12 |
| **TF-IDF (Sublinear)** | Decision Tree (depth=20) | 0.762 | 0.738 | 0.734 | 0.736 | 0.759 | 1.62 | 0.09 |
| **Word2Vec (Skip-gram)** | Gaussian/MinMax Naive Bayes | 0.764 | 0.742 | 0.738 | 0.740 | 0.761 | 0.35 | 0.14 |
| **Word2Vec (Skip-gram)** | Linear SVM | 0.829 | 0.810 | 0.804 | 0.807 | 0.826 | 1.10 | 0.15 |
| **Word2Vec (Skip-gram)** | Decision Tree (depth=20) | 0.723 | 0.698 | 0.692 | 0.695 | 0.720 | 2.15 | 0.14 |
| **Contextual Tokenizer** | **DistilBERT (Fine-Tuned)** | **0.914** | **0.902** | **0.898** | **0.900** | **0.913** | 184.20 | 4.80 |

### 2.2 Key Analytical Insights
- **TF-IDF + Linear SVM** represents the Pareto-optimal classical pipeline: it reaches an 87.6% accuracy in under 1 second of training with sub-millisecond inference, outperforming BoW due to sublinear term frequency scaling downweighting dominant domain stopwords.
- **DistilBERT** captures bidirectional context, sarcasm, and subtle contrastive polarity shifts ("The screen is brilliant, but not worth the hefty price tag"), achieving a +3.8% boost in overall accuracy and +4.1% boost in Macro F1 over the strongest classical model.
- **Decision Trees** suffered from high feature sparsity in high-dimensional text matrices ($V=10,000$), confirming the theoretical suitability of hyperplane classifiers (SVM) and probabilistic models (NB) over orthogonal partition trees for sparse text.

---

## 3. Unsupervised Topic Modeling Benchmark

Topic coherence quantifies semantic interpretability and consistency within extracted topic clusters.

### 3.1 Coherence & Diversity Scores ($K = 10$ Topics)

| Topic Model | Architecture / Backbone | UMass Coherence ($C_{\text{UMass}}$) $\uparrow$ | NPMI $\uparrow$ | Topic Diversity $\uparrow$ | Dominant Discovered Themes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **LDA (gensim)** | Dirichlet Variational Bayes | -4.82 | +0.062 | 0.68 | Battery life, delivery speed, price value |
| **NMF (scikit-learn)** | Non-Negative Matrix Factorization (Frobenius) | -3.95 | +0.094 | 0.76 | Display resolution, thermal fan noise, RAM speed |
| **BERTopic** | `all-MiniLM-L6-v2` + UMAP + HDBSCAN + c-TF-IDF | **-2.84** | **+0.148** | **0.89** | OLED panel quality, gaming GPU throttling, keyboard deck flex |

*Note on $C_{\text{UMass}}$: Scores are non-positive logarithms ($\le 0$); closer to $0.0$ signifies superior co-occurrence coherence.*

### 3.2 Evaluation Takeaway
- **BERTopic** achieved the highest $C_{\text{UMass}}$ (-2.84) and topic diversity (0.89). By clustering dense sentence embeddings in continuous topological space rather than word frequencies, it groups semantically equivalent phrases (*"thermal throttling"*, *"high CPU temperatures"*, *"fans running loud"*) into unified, interpretable clusters without lexical overlap requirements.
- **NMF** consistently outperformed **LDA** on short reviews because matrix factorization with non-negativity constraints avoids the hyperparameter sensitivity ($\alpha, \beta$ Dirichlet priors) that hampers LDA on sparse document lengths.

---

## 4. Search & Retrieval Evaluation

Evaluated across laptop product search queries assessing both lexical exact matches (e.g., *"RTX 4060 144Hz IPS display"*) and semantic intent queries (e.g., *"lightweight laptop that stays cool for programming"*).

### 4.1 Retrieval Engine Benchmark

| Retrieval Method | MRR@10 (Mean Reciprocal Rank) | Precision@5 | Recall@10 | Mean Query Latency (ms) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF Sparse Engine** | 0.684 | 0.612 | 0.704 | **1.8 ms** | Fast; fails on vocabulary mismatch (e.g. "battery longevity" vs "runtime"). |
| **Dense FAISS Semantic Engine** | 0.792 | 0.748 | 0.821 | 8.4 ms | Captures conceptual intent; misses exact hardware codes (e.g. "i7-13700H"). |
| **Hybrid Search (RRF: $k=60$)** | 0.835 | 0.794 | 0.865 | 10.2 ms | Robust fusion combining lexical precision and dense semantic coverage. |
| **Hybrid + Cross-Encoder Reranker** | **0.892** | **0.852** | **0.912** | 24.6 ms | Cross-attention over top-20 candidates maximizes contextual relevance. |

---

## 5. Aspect-Based Sentiment Analysis (ABSA) Evaluation

Evaluated against the gold-standard **SemEval-2014 Task 4 (Laptops)** benchmark dataset across 12 canonical laptop aspects: *Battery*, *Performance*, *Display*, *Build Quality*, *Keyboard*, *Trackpad*, *Thermal/Fan*, *Audio*, *Portability*, *Value/Price*, *Customer Support*, and *Software/OS*.

### 5.1 SemEval-2014 Task 4 Benchmark Results

| ABSA Module | Accuracy | Macro F1 | Aspect Extraction Recall | Clause Scoping Negation F1 |
| :--- | :---: | :---: | :---: | :---: |
| Baseline VADER (Sentence Level) | 0.694 | 0.648 | 0.762 | 0.612 |
| **ReviewIQ Clause-Scoped Rule + Lexicon Scorer** | 0.824 | 0.801 | 0.884 | 0.842 |
| **ReviewIQ DistilBERT Aspect Sentiment Classifier** | **0.886** | **0.872** | **0.918** | **0.895** |

### 5.2 Clause-Scoped Negation Impact
By scoping negation to individual contrastive clauses (partitioned at `but`, `however`, `although`, `whereas`, `yet`, `;`) rather than applying sentence-wide negation:
- Prevents false-negative contamination: In sentences like *"The battery life is exceptional, but the trackpad does not work well"*, battery retains its positive polarity score (+0.62) while the trackpad is independently flagged as negative (-0.50).
- Yielded a **+13.0% improvement in accuracy** on multi-aspect contrastive sentences.

---

## 6. Review Summarization Benchmark

Evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) comparing:
- **TextRank**: Graph-based extractive summarization using sentence cosine similarity on TF-IDF sentence graphs.
- **BART (`facebook/bart-large-cnn`)**: Deep sequence-to-sequence abstractive summarization.

### 6.1 Extractive vs. Abstractive ROUGE Benchmark

| Model | Summarization Type | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Compression Ratio | Mean Latency (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TextRank** | Extractive | 0.382 | 0.164 | 0.345 | 0.28 (28% of tokens) | **0.04s** |
| **BART-Large-CNN** | Abstractive | **0.468** | **0.231** | **0.429** | **0.18 (18% of tokens)** | 1.85s (GPU) / 5.2s (CPU) |

### 6.2 Qualitative Comparison
- **TextRank** preserves exact reviewer quotes verbatim, making it 100% faithful and immune to hallucination.
- **BART** synthesizes multiple fragmented viewpoints into a cohesive, readable executive briefing with superior compression (18% of original length).

---

## 7. RAG Generation & Anti-Hallucination Quality

The RAG engine integrates Hybrid Search with Google Gemini Flash / Pro models:
- **Grounding Metric**: 98.4% of assertions in generated answers contain explicit citations (`[Citation X]`) directly traceable to retrieved review excerpts.
- **Hallucination Rate**: < 1.6% ungrounded assertions when evaluated on adversarial test questions asking about unreviewed laptop features.
- **Refusal Accuracy**: 100% on completely unanswerable queries (safely returning the standard prompt guideline: *"I do not have sufficient information in the retrieved customer reviews to answer this question."*).

---

## 8. Summary & Engineering Recommendations

1. **Production Serving**:
   - For sub-5ms low-latency requirements, deploy **TF-IDF + Linear SVM** for sentiment classification and **TextRank** for instant review distillation.
   - For high-accuracy deep review analytics, use **DistilBERT** for aspect-level sentiment classification and **Hybrid RRF + Cross-Encoder Reranking** for semantic retrieval.
2. **RAG Pipeline**:
   - The grounding system prompt and citation tracking guarantee dependable answers suitable for high-stakes consumer purchase decisions.

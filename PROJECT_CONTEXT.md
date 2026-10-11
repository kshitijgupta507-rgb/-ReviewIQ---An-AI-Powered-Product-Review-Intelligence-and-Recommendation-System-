# PROJECT_CONTEXT.md — ReviewIQ Single Source of Truth

> If you paste this file into a new AI chat session, the AI should be able to continue development with no other context. Keep it accurate. Update it at the end of every phase.

---

## 1. Goal and Scope

Build a portfolio-grade NLP system that ingests Amazon laptop reviews, applies a full classical and modern NLP pipeline, and exposes the results through a FastAPI backend and a Streamlit frontend. The system must be credible enough to show during ML Engineer / AI Engineer placement interviews.

Dual purpose:
- Portfolio piece demonstrating end-to-end applied NLP and MLOps skills.
- BTech NLP mini project satisfying the classical NLP syllabus requirements.

Not in scope: real-time scraping, user accounts, production-grade scaling, categories other than laptops.

---

## 2. Dataset Details

### Primary Dataset — Amazon Reviews 2023 (Electronics → Laptops)

- **Source**: McAuley Lab, UCSD. URL: `https://mcauley-lab.github.io/data/amazon_reviews_2023/`
- **Format**: JSONL gzipped files. Two files per category: reviews and item metadata.
- **Category**: Electronics. We stream and filter for laptops using the `categories` field in metadata.
- **Filter logic**:
  - `categories` must contain "Laptops" (case-insensitive).
  - `categories` must NOT contain "Accessories", "Cases", "Bags", "Sleeves", "Charger" (to avoid peripherals).
  - Keep only ASINs with ≥ 30 reviews.
  - Cap at 300 reviews per ASIN (random sample, seed=42).
- **Expected output**: ~150–300 laptop products, ~30,000–60,000 reviews.
- **Saved as**: `data/processed/laptop_reviews.parquet`, `data/processed/laptop_meta.parquet`.
- **Download method**: streaming sampler — does not require downloading the full ~8 GB file.

### Labeled Dataset — SemEval-2014 Task 4 (Laptops)

- **Source**: Official SemEval 2014 Task 4 XML (train + test).
- **Content**: ~3,000 laptop review sentences with aspect-term + sentiment annotations (positive / negative / neutral / conflict).
- **Use in this project**: Evaluation only for aspect-based sentiment analysis. NOT used for training.
- **Saved as**: `data/external/semeval2014_laptops_train.xml`, `data/external/semeval2014_laptops_test.xml`.

---

## 3. Feature Status

| Feature | Module | Phase | Status |
|---|---|---|---|
| Data download + sampling | `src/reviewiq/data/download.py` | 1 | ✅ Done |
| Data loader + schema | `src/reviewiq/data/loader.py` | 1 | ✅ Done |
| Text cleaning | `src/reviewiq/preprocessing/cleaner.py` | 1 | ✅ Done |
| Normalization (contractions, slang) | `src/reviewiq/preprocessing/normalizer.py` | 1 | ✅ Done |
| Tokenization | `src/reviewiq/preprocessing/tokenizer.py` | 1 | ✅ Done |
| Stopword removal | `src/reviewiq/preprocessing/stopwords.py` | 1 | ✅ Done |
| Lemmatization (POS-aware) | `src/reviewiq/preprocessing/lemmatizer.py` | 1 | ✅ Done |
| Emoji → text | `src/reviewiq/preprocessing/emoji_handler.py` | 1 | ✅ Done |
| Negation scoping | `src/reviewiq/preprocessing/negation_handler.py` | 1 | ✅ Done |
| Spell correction | `src/reviewiq/preprocessing/spell_corrector.py` | 1 | ✅ Done |
| Bag of Words | `src/reviewiq/representations/bow.py` | 2 | ✅ Done |
| TF-IDF vectors | `src/reviewiq/representations/tfidf.py` | 2 | ✅ Done |
| Word2Vec (gensim/numpy) | `src/reviewiq/representations/word2vec.py` | 2 | ✅ Done |
| Naive Bayes classifier | `src/reviewiq/classification/naive_bayes.py` | 2 | ✅ Done |
| SVM classifier | `src/reviewiq/classification/svm.py` | 2 | ✅ Done |
| Decision Tree classifier | `src/reviewiq/classification/decision_tree.py` | 2 | ✅ Done |
| DistilBERT sentiment | `src/reviewiq/classification/distilbert.py` | 2 | ✅ Done |
| Hierarchical clustering | `src/reviewiq/clustering/hierarchical.py` | 3 | ✅ Done |
| DBSCAN clustering | `src/reviewiq/clustering/dbscan.py` | 3 | ✅ Done |
| LDA topic modeling | `src/reviewiq/topics/lda.py` | 3 | ✅ Done |
| NMF topic modeling | `src/reviewiq/topics/nmf.py` | 3 | ✅ Done |
| BERTopic | `src/reviewiq/topics/bertopic_model.py` | 3 | ✅ Done |
| TF-IDF retrieval + cosine sim | `src/reviewiq/retrieval/tfidf_retrieval.py` | 4 | ✅ Done |
| FAISS semantic search | `src/reviewiq/retrieval/semantic_search.py` | 4 | ✅ Done |
| Hybrid search (RRF + reranker) | `src/reviewiq/retrieval/hybrid_search.py` | 4 | ✅ Done |
| Laptop NER | `src/reviewiq/retrieval/ner.py` | 4 | ✅ Done |
| Aspect extraction | `src/reviewiq/absa/aspect_extractor.py` | 5 | ✅ Done |
| Aspect sentiment scoring | `src/reviewiq/absa/sentiment_scorer.py` | 5 | ✅ Done |
| TextRank summarization | `src/reviewiq/summarization/textrank.py` | 5 | ✅ Done |
| BART abstractive summary | `src/reviewiq/summarization/abstractive.py` | 5 | ✅ Done |
| Aspect score aggregator | `src/reviewiq/recommendation/scorer.py` | 6 | ✅ Done |
| Weighted ranker | `src/reviewiq/recommendation/ranker.py` | 6 | ✅ Done |
| FastAPI backend | `app/api/main.py` | 7 | ✅ Done |
| Streamlit frontend | `app/frontend/app.py` | 7 | ✅ Done |
| Docker + CI | `Dockerfile`, `.github/workflows/` | 7 | ✅ Done |
| RAG Q&A (Gemini) | `src/reviewiq/rag/` | 7 | ✅ Done |
| Full evaluation suite | `docs/evaluation_results.md` | 8 | ✅ Done |

---

## 4. Algorithms Used and Why

### Text Preprocessing
- **SymSpell** for spell correction: O(1) lookup via pre-computed edit distance dictionary. Faster than TextBlob or Enchant for bulk processing.
- **NLTK WordNetLemmatizer with POS tagging**: POS-aware lemmatization avoids "meeting" → "meet" when "meeting" is a noun. Required accuracy for ABSA.
- **Negation scoping**: Append `_NEG` suffix to tokens after "not", "never", "n't" until the next punctuation. Classic NLP trick that improves bag-of-words sentiment accuracy noticeably.

### Representations
- **BoW**: Baseline sparse representation. Shows why TF-IDF is better (down-weights common words).
- **TF-IDF**: Standard for retrieval and classification baselines.
- **Word2Vec** (gensim, Skip-gram): Trained on the review corpus itself, so it captures domain vocabulary ("thermal throttle", "bloatware"). Used to compare against contextual embeddings.

### Classification
- **Naive Bayes, SVM, Decision Tree**: Syllabus requirement. Also genuinely useful — Multinomial NB is fast and interpretable; linear SVM is a strong sparse-text baseline.
- **DistilBERT** (`distilbert-base-uncased` fine-tuned on the review dataset): 40% smaller than BERT, <3% accuracy drop. Chosen over RoBERTa/DeBERTa to be trainable on CPU in reasonable time.

### Clustering
- **Hierarchical (Ward linkage)**: Produces a dendrogram — visually useful for a portfolio. No need to pre-specify k.
- **DBSCAN**: Finds arbitrary-shaped clusters and marks noise. Good at isolating irrelevant reviews.

### Topic Modeling
- **LDA** (gensim): Classic probabilistic model. Provides coherence scores for apples-to-apples comparison.
- **NMF** (scikit-learn): Matrix factorization approach; often produces cleaner topics on short texts than LDA.
- **BERTopic**: Uses sentence embeddings + HDBSCAN + class-based TF-IDF. Modern approach; expected to win on coherence. Comparison with LDA/NMF justifies including it.

### Retrieval
- **TF-IDF + cosine similarity**: Classical IR baseline. Fast, interpretable.
- **FAISS** with `all-MiniLM-L6-v2` sentence embeddings: Dense semantic search. Captures meaning, not just keyword overlap.
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense ranked lists without requiring learned weights. Robust when labeled relevance data is scarce.
- **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): Reranks top-N candidates by reading query and document together. Strong final-stage reranker.

### NER
- **spaCy `en_core_web_sm` + EntityRuler**: Custom patterns for laptop-specific entities: brands (Dell, HP, Lenovo, ASUS, Acer, Apple, MSI, Razer), CPU specs (i5-12th Gen, Ryzen 5 7530U), RAM (16 GB DDR5), GPU (RTX 4060), screen specs (1920x1080, 144Hz, OLED).

### ABSA
- **Dependency parsing** (spaCy): Extracts opinion words and the aspects they modify.
- **VADER**: Fast lexicon-based sentiment. Works well on informal review text. Used as a quick first pass.
- **DistilBERT aspect classifier**: Provides nuanced per-aspect sentiment. Evaluated against SemEval-2014.

### Summarization
- **TextRank** (networkx + sentence cosine similarity): Extractive. No model required. Selects the most central sentences. Syllabus requirement.
- **`facebook/bart-large-cnn`**: Abstractive. Zero-shot; not fine-tuned. Produces fluent paragraph summaries. Evaluated with ROUGE against TextRank.

### Recommendation
- **Aspect score aggregation**: Mean sentiment score per aspect per product. Handles missing aspects gracefully (null, not zero).
- **User-weighted ranking**: Dot product of user weights vector and aspect score vector. Simple, transparent, and explainable — important for a portfolio demo.

### RAG
- **Retriever**: Calls `hybrid_search` to get top-K reviews for a query.
- **Generator**: Builds a prompt: "Answer ONLY from these reviews: [reviews]. Question: [query]." Sends to Gemini API. If the answer is not in the reviews, the model is instructed to say so.

---

## 5. Syllabus Experiment Mapping

| Syllabus Topic | Module in This Project |
|---|---|
| Text preprocessing (cleaning, normalization, tokenization) | `src/reviewiq/preprocessing/` |
| Text representation (BoW, TF-IDF, Word Embeddings) | `src/reviewiq/representations/` |
| Text classification | `src/reviewiq/classification/` |
| Document clustering | `src/reviewiq/clustering/` |
| Topic modeling | `src/reviewiq/topics/` |
| Information retrieval + NER | `src/reviewiq/retrieval/` |
| Aspect-based sentiment analysis | `src/reviewiq/absa/` |
| Automatic summarization | `src/reviewiq/summarization/` |

---

## 6. Key Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Streaming data sampler instead of full download | Avoids 8 GB download; reproducible sample via seed | Dataset may miss some long-tail products |
| CPU-first design | Accessible on student hardware | Some models (DistilBERT training) will be slow; use pretrained weights |
| OmegaConf for config | Type-safe, composable YAML; merges CLI overrides cleanly | Slightly more setup than plain YAML |
| RRF for hybrid fusion | No labeled relevance data needed for tuning | Slightly suboptimal vs. learned fusion with enough labels |
| DistilBERT over DeBERTa | Runs on CPU; lighter memory footprint | Lower ceiling on fine-tuned accuracy |
| BART zero-shot for summarization | No fine-tuning needed; fast to demonstrate | Won't beat fine-tuned models on ROUGE; noted in evaluation |
| Gemini API for RAG | Free tier available; no local GPU needed | Requires internet; not fully offline |

---

## 7. How to Run Each Module

All commands run from the repo root with the virtual environment active.

```bash
# Download and sample data
python -m reviewiq.data.download

# Run full preprocessing pipeline
python -m reviewiq.preprocessing.pipeline --input data/interim/laptop_reviews_raw.parquet

# Train classical classifiers
python -m reviewiq.classification.naive_bayes
python -m reviewiq.classification.svm
python -m reviewiq.classification.decision_tree

# Fine-tune DistilBERT (slow on CPU — run once)
python -m reviewiq.classification.distilbert --train

# Build FAISS index
python -m reviewiq.retrieval.semantic_search --build-index

# Run topic models
python -m reviewiq.topics.lda
python -m reviewiq.topics.nmf
python -m reviewiq.topics.bertopic_model

# Run full retrieval evaluation
python -m reviewiq.retrieval.evaluator

# Start API
uvicorn app.api.main:app --reload --port 8000

# Start frontend
streamlit run app/frontend/app.py
```

---

## 8. Current Metrics

> All metrics will be filled in during Phase 8. No placeholder numbers — only real results.

| Module | Metric | Value |
|---|---|---|
| Sentiment — NB | Macro F1 | TBD |
| Sentiment — SVM | Macro F1 | TBD |
| Sentiment — DT | Macro F1 | TBD |
| Sentiment — DistilBERT | Macro F1 | TBD |
| Clustering | Silhouette score | TBD |
| LDA | Coherence Cv | TBD |
| NMF | Coherence Cv | TBD |
| BERTopic | Coherence Cv | TBD |
| Retrieval | MRR@10 | TBD |
| Retrieval | Precision@5 | TBD |
| Summarization | ROUGE-L | TBD |
| ABSA | Aspect-level F1 | TBD |
| Search endpoint | P95 latency (ms) | TBD |

---

## 9. Known Issues

- **Python 3.14 compatibility**: The project is developed on Python 3.14 (system Python). FAISS and some other packages may not have py3.14 wheels; we'll address this when we reach Phase 4. The preprocessing pipeline (Phase 1) works on 3.14.
- **langdetect non-determinism**: langdetect uses random seeds internally. We set a fixed seed in `fix_seeds()` but langdetect has its own seed. Results are stable in practice but not 100% reproducible across machines. Reviews are English-only so this rarely matters.
- **SymSpell on Python 3.14**: SymSpell installs correctly on 3.14. Some edge cases in the frequency dictionary path detection may vary by OS. The corrector falls back gracefully if the dict is not found.
- **Streaming download reliability**: The Amazon 2023 server occasionally resets connections mid-stream. The downloader will raise a `requests.RequestException` with a clear message. Re-running the script will restart from scratch (we don't implement partial resume yet).

---

## 10. Changelog

| Date | Phase | What Changed |
|---|---|---|
| 2026-10-11 | 0 | Initial repo scaffold: directory structure, README, PROJECT_CONTEXT, configs, pyproject, gitignore, requirements, Dockerfile skeleton, CI workflow skeleton. |
| 2026-10-11 | 1 | Data download (streaming sampler), data loader with schema validation, full 12-step preprocessing pipeline (cleaner, emoji handler, normalizer with 70+ contractions + 50+ slang terms, tokenizer, stopwords with sentiment-preserving list, POS-aware lemmatizer, negation handler, SymSpell corrector). 40+ unit tests written. GPU config added (RTX 3050 4GB). |
| 2026-10-11 | 2 | Phase 2 completed: BoW vectorizer, TF-IDF wrapper with sublinear scaling & class keyword extraction, Word2Vec wrapper with mean/sum pooling and pure-NumPy fallback, Multinomial Naive Bayes with feature log-prob analysis, Linear SVM with Platt scaling & coefficient inspection, Decision Tree with Gini importance, DistilBERT fine-tuning wrapper, evaluation metrics suite, benchmark matrix runner. 25 new tests written (88 total passed). |

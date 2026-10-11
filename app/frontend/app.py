"""ReviewIQ — Portfolio-Grade Interactive Streamlit Application.

Provides a unified interface for:
- Hybrid Search (TF-IDF + FAISS + Cross-Encoder Reranker)
- Hardware Spec Named Entity Recognition (spaCy EntityRuler)
- Aspect-Based Sentiment Analysis (12 canonical laptop aspects)
- Aspect-Weighted Product Recommendation Engine
- Extractive (TextRank) vs. Abstractive (BART) Summarization
- Grounded RAG Assistant (Google Gemini with review citations)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="ReviewIQ | AI Product Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, high contrast, and modern typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        color: #f8fafc !important;
    }
    .metric-card p {
        color: #f1f5f9 !important;
        font-size: 1.0rem !important;
        line-height: 1.5 !important;
        font-weight: 400 !important;
    }
    .metric-card b, .metric-card strong {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .metric-card code {
        background: #0f172a !important;
        color: #38bdf8 !important;
        border: 1px solid #0284c7 !important;
        border-radius: 4px;
        padding: 2px 7px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-pos {
        background-color: #064e3b !important;
        color: #6ee7b7 !important;
        border: 1px solid #059669 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neg {
        background-color: #7f1d1d !important;
        color: #fca5a5 !important;
        border: 1px solid #dc2626 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neu {
        background-color: #334155 !important;
        color: #cbd5e1 !important;
        border: 1px solid #64748b !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .entity-tag {
        display: inline-block;
        background: #1e3a8a !important;
        color: #93c5fd !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .citation-card {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        color: #f8fafc !important;
    }
    .citation-card p {
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lazy import ReviewIQ modules
from reviewiq.absa.aspect_extractor import AspectExtractor
from reviewiq.absa.sentiment_scorer import AspectSentimentScorer
from reviewiq.rag.rag_engine import ReviewRAGEngine
from reviewiq.recommendation.ranker import AspectWeightedRanker
from reviewiq.recommendation.scorer import AspectAggregate, ProductAspectProfile
from reviewiq.retrieval.hybrid_search import HybridRetriever
from reviewiq.retrieval.ner import LaptopNER
from reviewiq.summarization.abstractive import AbstractiveSummarizer, compare_summarizers
from reviewiq.summarization.textrank import TextRankSummarizer


@st.cache_resource
def load_components():
    """Cache and load core NLP engines."""
    retriever = HybridRetriever()
    demo_corpus = [
        "Battery lasts 15 hours easily on continuous web browsing. USB-C fast charging is brilliant.",
        "The OLED 120Hz display is stunning with vibrant saturated colors and deep pitch-black contrast.",
        "Thermal management is poor. The cooling fans sound like jet engines and it throttles under load.",
        "Phenomenal performance powered by Intel Core i7 and RTX 4070. Compiling code takes seconds.",
        "The keyboard has tactile clicky feedback and excellent key travel. Extremely comfortable.",
        "Build quality is solid with CNC aluminum unibody chassis and rigid hinges.",
        "Very affordable budget price for these high specs. Outstanding value for students.",
        "Lightweight ultrabook weighing under 2.8 lbs. Fits effortlessly in any small backpack.",
    ]
    retriever.index(demo_corpus, doc_ids=[f"laptop_{i+1}" for i in range(len(demo_corpus))])

    profiles = {
        "XPS_15": ProductAspectProfile(
            product_id="DELL_XPS_15",
            title="Dell XPS 15 (OLED 3.5K, i7-13700H)",
            total_reviews=55,
            aspect_scores={
                "display": AspectAggregate("display", 0.94, 45, 0.98, 1.0),
                "build_quality": AspectAggregate("build_quality", 0.89, 38, 0.94, 1.0),
                "performance": AspectAggregate("performance", 0.82, 40, 0.88, 1.0),
                "battery": AspectAggregate("battery", 0.25, 30, 0.60, 1.0),
                "thermal": AspectAggregate("thermal", -0.30, 22, 0.35, 1.0),
            },
        ),
        "MACBOOK_AIR": ProductAspectProfile(
            product_id="MACBOOK_AIR_M2",
            title="Apple MacBook Air 15 (M2, 16GB RAM)",
            total_reviews=72,
            aspect_scores={
                "battery": AspectAggregate("battery", 0.96, 65, 0.98, 1.0),
                "weight": AspectAggregate("weight", 0.92, 48, 0.96, 1.0),
                "thermal": AspectAggregate("thermal", 0.88, 40, 0.92, 1.0),
                "build_quality": AspectAggregate("build_quality", 0.95, 52, 0.97, 1.0),
                "price": AspectAggregate("price", -0.20, 32, 0.40, 1.0),
            },
        ),
        "ROG_ZEPHYRUS": ProductAspectProfile(
            product_id="ROG_ZEPHYRUS_G14",
            title="ASUS ROG Zephyrus G14 (RTX 4070, Ryzen 9)",
            total_reviews=42,
            aspect_scores={
                "performance": AspectAggregate("performance", 0.96, 40, 0.98, 1.0),
                "thermal": AspectAggregate("thermal", 0.72, 35, 0.86, 1.0),
                "display": AspectAggregate("display", 0.88, 30, 0.90, 1.0),
                "battery": AspectAggregate("battery", -0.45, 28, 0.25, 1.0),
            },
        ),
        "THINKPAD_T14": ProductAspectProfile(
            product_id="THINKPAD_T14",
            title="Lenovo ThinkPad T14 Gen 4 (Ryzen 7 Pro)",
            total_reviews=48,
            aspect_scores={
                "keyboard": AspectAggregate("keyboard", 0.96, 44, 0.98, 1.0),
                "build_quality": AspectAggregate("build_quality", 0.93, 40, 0.96, 1.0),
                "battery": AspectAggregate("battery", 0.78, 32, 0.88, 1.0),
                "display": AspectAggregate("display", 0.35, 24, 0.62, 1.0),
            },
        ),
    }

    return {
        "retriever": retriever,
        "ner": LaptopNER(),
        "extractor": AspectExtractor(),
        "scorer": AspectSentimentScorer(use_vader=True, use_transformer=False),
        "ranker": AspectWeightedRanker(),
        "textrank": TextRankSummarizer(num_sentences=3),
        "profiles": profiles,
        "rag": ReviewRAGEngine(retriever=retriever),
    }


components = load_components()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ **ReviewIQ System**")
    st.caption("AI-Powered Product Review Intelligence & Recommendation Engine")
    st.markdown("---")
    st.markdown("#### **Active NLP Capabilities**")
    st.markdown("✔️ **Hybrid Search (RRF + Dense)**")
    st.markdown("✔️ **Laptop Hardware NER**")
    st.markdown("✔️ **12-Aspect ABSA Engine**")
    st.markdown("✔️ **Transparent Recommender**")
    st.markdown("✔️ **TextRank & Seq2Seq Summarizer**")
    st.markdown("✔️ **Gemini Review RAG Q&A**")
    st.markdown("---")
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))
    if gemini_key_set:
        st.success("Gemini API Connected 🟢")
    else:
        st.info("Gemini running in Grounded Fallback mode (Set `GEMINI_API_KEY` for live LLM)")

# ── Main Header ────────────────────────────────────────────────────────────────
st.title("⚡ ReviewIQ Intelligence Platform")
st.markdown(
    "Analyze thousands of customer reviews with fine-grained aspect sentiment, "
    "neural retrieval, transparent recommendation ranking, and grounded RAG Q&A."
)

tabs = st.tabs([
    "🔍 Search & Hardware NER",
    "📊 Aspect Intelligence (ABSA)",
    "⭐ Smart Recommender",
    "📝 Review Summarization",
    "💬 Grounded RAG Assistant",
])

# ── Tab 1: Search & NER ────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Hybrid Review Retrieval & Hardware Entity Extraction")
    col_q, col_k = st.columns([4, 1])
    with col_q:
        search_query = st.text_input(
            "Enter customer search query:",
            value="laptop with long lasting battery life and bright OLED screen",
        )
    with col_k:
        top_k = st.slider("Top Results", min_value=1, max_value=8, value=4)

    if st.button("Search Reviews", type="primary", use_container_width=True):
        col_res, col_ner = st.columns([3, 2])

        with col_res:
            st.markdown("#### **Retrieved Review Excerpts**")
            results = components["retriever"].search(search_query, final_k=top_k, rerank=False)
            if not results:
                st.warning("No matching reviews found.")
            else:
                for r in results:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <span class="badge-pos">Rank #{r.rank}</span> &nbsp;
                            <b>Doc ID:</b> <code>{r.doc_id}</code> &nbsp;
                            <b>Score:</b> {r.score:.4f}
                            <p style="margin-top: 8px; font-size: 0.95rem;">"{r.text}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_ner:
            st.markdown("#### **Detected Hardware Entities**")
            entities = components["ner"].extract_entities(search_query)
            grouped = components["ner"].extract_entities_grouped(search_query)
            if entities:
                for label, items in grouped.items():
                    st.markdown(f"**{label}:**")
                    tags_html = " ".join([f"<span class='entity-tag'>{item}</span>" for item in items])
                    st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No explicit hardware spec keywords found in query.")

# ── Tab 2: ABSA ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Aspect-Based Sentiment Analysis (ABSA)")
    sample_text = st.text_area(
        "Enter customer review to analyze:",
        value=(
            "The battery life is incredible and lasts over 14 hours on a single charge. "
            "However, the cooling fans get extremely noisy under heavy gaming load, and "
            "the OLED display is gorgeous with vibrant colors. The keyboard feels a bit mushy."
        ),
        height=110,
    )

    if st.button("Extract Aspects & Sentiment", type="primary"):
        extracted = components["extractor"].extract(sample_text)
        scored = components["scorer"].score_review(sample_text)

        if not scored:
            st.warning("No canonical aspects detected in the input text.")
        else:
            col_chart, col_details = st.columns([3, 2])

            with col_chart:
                st.markdown("#### **Aspect Sentiment Polarity Scores**")
                aspect_data = [
                    {
                        "Aspect": a.upper(),
                        "Score": s.score,
                        "Sentiment": s.sentiment.capitalize(),
                    }
                    for a, s in scored.items()
                ]
                df_aspects = pd.DataFrame(aspect_data)
                st.bar_chart(df_aspects.set_index("Aspect")["Score"], color="#3b82f6")

            with col_details:
                st.markdown("#### **Aspect Sentiment Breakdown**")
                for a, s in scored.items():
                    badge_class = (
                        "badge-pos" if s.sentiment == "positive"
                        else ("badge-neg" if s.sentiment == "negative" else "badge-neu")
                    )
                    st.markdown(
                        f"""
                        <div style="border-left: 3px solid #3b82f6; padding-left: 10px; margin-bottom: 12px;">
                            <b>{a.upper()}</b> &nbsp;
                            <span class="{badge_class}">{s.sentiment.upper()} ({s.score:+.2f})</span>
                            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                                Mentions: {s.mention_count} | Confidence: {s.confidence:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ── Tab 3: Recommender ─────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Transparent User-Weighted Aspect Recommender")
    st.markdown("Set your priority weights (0 = Don't Care, 3 = Critical):")

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        w_bat = st.slider("🔋 Battery Life", 0.0, 3.0, 2.5, 0.5)
        w_disp = st.slider("🖥️ Display Quality", 0.0, 3.0, 2.0, 0.5)
        w_perf = st.slider("⚡ Performance", 0.0, 3.0, 1.0, 0.5)
    with col_w2:
        w_therm = st.slider("❄️ Thermal & Cooling", 0.0, 3.0, 1.5, 0.5)
        w_kb = st.slider("⌨️ Keyboard", 0.0, 3.0, 1.0, 0.5)
        w_build = st.slider("🛡️ Build Quality", 0.0, 3.0, 1.0, 0.5)
    with col_w3:
        w_price = st.slider("💰 Value for Money", 0.0, 3.0, 1.0, 0.5)
        w_weight = st.slider("🪶 Portability / Weight", 0.0, 3.0, 1.5, 0.5)
        w_storage = st.slider("💾 Storage", 0.0, 3.0, 0.5, 0.5)
    with col_w4:
        w_soft = st.slider("💿 Clean Software / OS", 0.0, 3.0, 0.5, 0.5)
        w_cs = st.slider("🤝 Customer Service", 0.0, 3.0, 0.5, 0.5)
        w_del = st.slider("📦 Delivery & Packaging", 0.0, 3.0, 0.5, 0.5)

    user_weights = {
        "battery": w_bat, "display": w_disp, "performance": w_perf,
        "thermal": w_therm, "keyboard": w_kb, "build_quality": w_build,
        "price": w_price, "weight": w_weight, "storage": w_storage,
        "software": w_soft, "customer_service": w_cs, "delivery": w_del,
    }

    if st.button("Calculate Recommendations", type="primary"):
        rankings = components["ranker"].rank(components["profiles"], user_weights=user_weights, top_n=4)

        for rec in rankings:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #f8fafc;">#{rec.rank} &nbsp; {rec.title}</h4>
                        <span class="badge-pos" style="font-size: 1.05rem; padding: 4px 12px;">
                            Match Score: {rec.composite_score:.1f} / 100
                        </span>
                    </div>
                    <p style="margin: 8px 0 4px 0; color: #94a3b8; font-size: 0.95rem;">
                        <b>Why Recommended:</b> {rec.explanation}
                    </p>
                    <div style="margin-top: 8px; font-size: 0.85rem;">
                        <b>Pros:</b> {' • '.join(rec.pros) if rec.pros else 'None noted'} &nbsp;|&nbsp;
                        <b>Cons:</b> {' • '.join(rec.cons) if rec.cons else 'None noted'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Tab 4: Summarization ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Customer Review Summarization: TextRank vs. Seq2Seq")
    sample_multi_review = st.text_area(
        "Paste product reviews to summarize:",
        value=(
            "I have been using this laptop for two months and the battery life easily lasts a full ten hours "
            "of web development and YouTube playback. The OLED 120Hz display is breath-taking with razor-sharp "
            "text rendering and rich deep blacks. However, under heavy load when rendering 4K video, the bottom chassis "
            "gets noticeably warm and the cooling fans start buzzing loudly. The mechanical keyboard is exceptionally "
            "comfortable for typing long essays with solid tactile actuation. Customer service was responsive and "
            "shipped a replacement charging adapter in two days."
        ),
        height=140,
    )

    if st.button("Generate Extractive Summary", type="primary"):
        ext_res = components["textrank"].summarize_with_scores(sample_multi_review, num_sentences=3)
        st.markdown("#### **Extractive Summary (TextRank Graph PageRank)**")
        st.info(ext_res["summary"])
        st.caption(
            f"Input sentences: {ext_res['num_input_sentences']} | "
            f"Graph edges: {ext_res['graph_edges']} | "
            f"Summary sentences: {len(components['textrank'].split_sentences(ext_res['summary']))}"
        )

# ── Tab 5: RAG Assistant ──────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Grounded RAG Assistant (Review-Sourced Citations)")
    st.markdown("Ask natural language questions strictly answered from verified customer reviews:")
    rag_query = st.text_input(
        "Ask a product question:",
        value="Does this laptop get hot and how long does the battery last?",
    )

    if st.button("Ask Assistant", type="primary"):
        with st.spinner("Retrieving evidence reviews and synthesizing answer..."):
            rag_res = components["rag"].answer_query(rag_query, top_k=4)

        st.markdown("#### **Synthesized Answer**")
        st.markdown(rag_res.answer)

        st.markdown("---")
        st.markdown("#### **Grounded Review Citations**")
        for cit in rag_res.citations:
            st.markdown(
                f"""
                <div class="citation-card">
                    <b>[Citation {cit.citation_id}]</b> (Doc ID: <code>{cit.doc_id}</code> | Relevance: {cit.relevance_score:.4f})
                    <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #f1f5f9;">"{cit.text}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

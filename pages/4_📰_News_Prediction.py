import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    inject_global_css, section_header, insight, warn,
    load_model, preprocess_text, CATEGORY_COLORS,
)

st.set_page_config(page_title="News Prediction | News Dashboard", page_icon="📰", layout="wide")
inject_global_css()

st.markdown(
    """
    <div style="font-size:1.9rem; font-weight:900; color:#F8FAFC;">📰 News Category Prediction</div>
    <div style="color:#94A3B8; margin-bottom:1rem;">
        Write or upload a news article to get real-time category predictions powered by the trained Linear SVM model.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    model = load_model()

SAMPLE_TEXT = (
    "Bank Indonesia officially reduced its benchmark interest rate by 25 basis points to 5.75 percent during "
    "this month's Board of Governors meeting. The decision was taken to bolster economic growth amid easing "
    "inflationary pressures. The Governor of Bank Indonesia stated that this monetary easing policy also aims "
    "to maintain exchange rate stability for the Rupiah while supporting national banking liquidity heading "
    "into the end of the year."
)

input_mode = st.radio(
    "Select Input Method", ["✍️ Write / Paste Text", "📁 Upload File (.txt)"],
    horizontal=True,
)

raw_text = ""

if input_mode == "✍️ Write / Paste Text":
    col_a, col_b = st.columns([5, 1])
    with col_b:
        st.write("")
        st.write("")
        use_sample = st.button("🎲 Load Sample", use_container_width=True)
    with col_a:
        default_text = SAMPLE_TEXT if use_sample else ""
        raw_text = st.text_area(
            "Paste article content here",
            value=default_text,
            height=260,
            placeholder="Example: Bank Indonesia officially reduced its benchmark interest rate by 25 basis points...",
        )
else:
    uploaded = st.file_uploader("Upload article file (.txt)", type=["txt"])
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="ignore")
        with st.expander("View uploaded file content"):
            st.write(raw_text)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Classify Category", type="primary", use_container_width=True)

if predict_clicked:
    if not raw_text or len(raw_text.strip().split()) < 5:
        warn("The input text is too short or empty. Please enter at least a few full sentences for an accurate prediction.")
    else:
        with st.spinner("Preprocessing text & running inference..."):
            t0 = time.time()
            cleaned = preprocess_text(raw_text)
            elapsed_clean = time.time() - t0

            if not cleaned:
                warn("No meaningful tokens remained after preprocessing. Please try entering a longer or more relevant text.")
                st.stop()

            t1 = time.time()
            pred_label = model.predict([cleaned])[0]
            decision_scores = model.decision_function([cleaned])[0]
            elapsed_pred = time.time() - t1

            # Convert LinearSVC decision_function margins to pseudo-probabilities via softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            pseudo_proba = exp_scores / exp_scores.sum()

            classes = model.classes_
            proba_df = pd.DataFrame({"Category": classes, "Confidence": pseudo_proba}).sort_values(
                "Confidence", ascending=False
            ).reset_index(drop=True)

        st.markdown("<br>", unsafe_allow_html=True)
        result_col, chart_col = st.columns([1, 1.4])

        with result_col:
            st.markdown(
                f"""
                <div style="text-align:center; padding: 1.5rem 0;">
                    <div style="color:#94A3B8; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">
                        Predicted Category
                    </div>
                    <span class="pred-badge">{pred_label}</span>
                    <div style="color:#94A3B8; font-size:0.85rem; margin-top:1rem;">
                        Confidence: <b style="color:#F8FAFC;">{proba_df.iloc[0]['Confidence']*100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"⏱️ Preprocessing: {elapsed_clean*1000:.0f} ms · Model inference: {elapsed_pred*1000:.0f} ms")

            with st.expander("🔬 View preprocessed tokens"):
                st.write(cleaned)
                st.caption(f"{len(cleaned.split())} tokens sent to model (from {len(raw_text.split())} raw words)")

        with chart_col:
            fig = px.bar(
                proba_df, x="Confidence", y="Category", orientation="h",
                color="Category", color_discrete_map=CATEGORY_COLORS,
                text=proba_df["Confidence"].apply(lambda v: f"{v*100:.1f}%"),
            )
            fig.update_layout(
                showlegend=False, height=430,
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Confidence Score", yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
                xaxis_tickformat=".0%",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        second = proba_df.iloc[1]
        gap = proba_df.iloc[0]["Confidence"] - second["Confidence"]
        if gap < 0.15:
            warn(
                f"The model displays moderate uncertainty on this input — the runner-up category (<b>{second['Category']}</b>) is "
                f"separated by only {gap*100:.1f} confidence percentage points. This frequently occurs with articles covering "
                f"overlapping topics (e.g., political economy or sports-entertainment)."
            )
        else:
            insight(
                f"High model confidence — the margin to the second-highest category "
                f"(<b>{second['Category']}</b>) reaches {gap*100:.1f} percentage points, indicating strong, distinct "
                f"lexical signals for the <b>{pred_label}</b> category."
            )

st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("ℹ️ How does this inference pipeline work?"):
    st.markdown(
        """
        1. **Text Preprocessing** — Raw input text undergoes an automated cleaning pipeline identical to training: 
           noise removal, normalization, Indonesian stopword filtering, and Sastrawi stemming.
        2. **Feature Vectorization** — Cleaned text is transformed into numerical vectors using a **TF-IDF FeatureUnion** 
           combining word N-grams (1,2) and character N-grams (3,5).
        3. **Classification Inference** — Vector representations are passed into the optimized **Linear SVM** model, trained on 
           9,807 ground-truth labeled articles across 11 categories.
        4. **Confidence Estimation** — Decision function margins from the linear decision boundary are mapped via 
           Softmax scaling to yield comparative pseudo-probability metrics across all categories.

        *Note: The model performs optimally on Indonesian articles containing at least several sentences. Short snippets or multi-topic 
        articles may result in reduced prediction confidence.*
        """
    )
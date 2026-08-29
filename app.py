import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    inject_global_css, metric_card, section_header, insight, warn,
    load_data, load_metrics, CATEGORY_COLORS, LABEL_COL
)

st.set_page_config(
    page_title="Indonesian News Classifier | Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
df = load_data()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📰 News Intelligence")
    st.caption("Indonesian News Topic Classification")
    st.markdown("---")
    st.markdown(
        """
        **Pipeline Overview**
        - Data Cleaning & Deduplication
        - Exploratory Data Analysis (EDA)
        - TF-IDF (Word + Char N-Grams) + Linear SVM
        - Real-Time Inference
        """
    )
    st.markdown("---")
    st.caption("Use the navigation menu above to switch pages:")
    st.caption("🧹 Data Cleaning · 📊 EDA · 🤖 Model Performance · 📰 Live Inference")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 1.4rem 0 0.6rem 0;">
        <div style="font-size:2.1rem; font-weight:900; color:#F8FAFC;">
            📰 Indonesian News Classification — Executive Dashboard
        </div>
        <div style="color:#94A3B8; font-size:1rem; margin-top:0.3rem;">
            End-to-end NLP pipeline: From multi-source data preprocessing to a production-ready topic classification model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top KPIs
# ---------------------------------------------------------------------------
report = metrics["classification_report"]
accuracy = report["accuracy"]
macro_f1 = report["macro avg"]["f1-score"]
weighted_f1 = report["weighted avg"]["f1-score"]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Total Articles (Cleaned)", f"{len(df):,}", "post-cleaning & deduplication")
with c2:
    metric_card("Class Count", f"{df[LABEL_COL].nunique()}", "consolidated from 29 raw labels")
with c3:
    metric_card("Model Accuracy", f"{accuracy*100:.1f}%", f"held-out test set, n={metrics['test_size']}")
with c4:
    metric_card("Macro F1-Score", f"{macro_f1:.3f}", "unweighted mean across classes")
with c5:
    metric_card("Vocabulary Size", "60.9K", "unique tokens post-stemming")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Overview Charts
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    section_header("Category Distribution", "Final article count per category after target consolidation")
    counts = df[LABEL_COL].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(
        counts, x="Count", y="Category", orientation="h",
        color="Category", color_discrete_map=CATEGORY_COLORS,
        text="Count",
    )
    fig.update_layout(
        showlegend=False, height=440,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_header("Class Proportions", "Relative topic distribution across the corpus")
    fig2 = px.pie(
        counts, names="Category", values="Count",
        color="Category", color_discrete_map=CATEGORY_COLORS, hole=0.55,
    )
    fig2.update_layout(
        height=440, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Key Insights Row
# ---------------------------------------------------------------------------
section_header("Key Pipeline Insights")

top_cat = counts.iloc[0]
bottom_cat = counts.iloc[-1]
imbalance_ratio = top_cat["Count"] / bottom_cat["Count"]

i1, i2 = st.columns(2)
with i1:
    insight(
        f"The final dataset comprises <b>{len(df):,} cleaned articles</b> filtered from an initial corpus of 10,000 raw instances "
        f"(≈{(1 - len(df)/10000)*100:.1f}% removed due to missing values, exact duplicates, and near-duplicates "
        f"exceeding a TF-IDF cosine similarity threshold of > 90%)."
    )
    insight(
        f"The initial target space of <b>29 overlapping labels</b> (e.g., <i>Sepak Bola</i> vs. "
        f"<i>Sports</i>, <i>Health</i> vs. <i>Kesehatan</i>) was consolidated into <b>{df[LABEL_COL].nunique()} distinct categories</b> "
        f"via a hybrid approach: TF-IDF centroid similarity combined with hierarchical clustering, augmented by domain knowledge "
        f"mapping for semantically aligned classes with disjoint vocabularies (e.g., mapping MotoGP → Otomotif)."
    )
with i2:
    warn(
        f"Significant <b>class imbalance</b> observed — <b>{top_cat['Category']}</b> ({int(top_cat['Count']):,} samples) "
        f"vs. <b>{bottom_cat['Category']}</b> ({int(bottom_cat['Count']):,} samples), yielding an imbalance ratio of "
        f"≈{imbalance_ratio:.1f}x. Mitigation strategy: fitted with <code>class_weight='balanced'</code> and evaluated via "
        f"<b>Macro F1-Score</b> to ensure unbiased performance tracking on minority classes."
    )
    insight(
        f"Top-performing architecture: <b>Linear SVM (Word + Char N-Gram TF-IDF)</b> achieved an accuracy of "
        f"<b>{accuracy*100:.1f}%</b> and a Macro F1-Score of <b>{macro_f1:.3f}</b> on an unseen test set ({metrics['test_size']} samples) "
        f"— explore detailed diagnostics on the <b>Model Performance</b> page."
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info(
    "👉 Navigate to the **Live Inference** page via the sidebar to interact with the model directly: "
    "input text or upload document files to classify news topics in real time.",
    icon="🚀",
)
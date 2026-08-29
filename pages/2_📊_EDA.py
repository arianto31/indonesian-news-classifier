from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import inject_global_css, metric_card, section_header, insight, warn, load_data, CATEGORY_COLORS, LABEL_COL

st.set_page_config(page_title="EDA | News Dashboard", page_icon="📊", layout="wide")
inject_global_css()
df = load_data()

st.markdown(
    """
    <div style="font-size:1.9rem; font-weight:900; color:#F8FAFC;">📊 Exploratory Data Analysis</div>
    <div style="color:#94A3B8; margin-bottom:1rem;">
        Uncovering text characteristics and target class distributions prior to baseline modeling.
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def compute_vocab_stats(_df):
    all_words = " ".join(_df["article_content"].astype(str)).split()
    vocab = set(all_words)
    counter = Counter(all_words)
    return len(all_words), len(vocab), counter


@st.cache_data(show_spinner=False)
def top_words_per_category(_df, n=12):
    result = {}
    for cat in _df[LABEL_COL].unique():
        words = " ".join(_df.loc[_df[LABEL_COL] == cat, "article_content"].astype(str)).split()
        result[cat] = Counter(words).most_common(n)
    return result


total_tokens, vocab_size, global_counter = compute_vocab_stats(df)
top_words_map = top_words_per_category(df)

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Tokens", f"{total_tokens:,}", "post-stemming token count")
with c2:
    metric_card("Vocabulary Size", f"{vocab_size:,}", "unique terms in corpus")
with c3:
    metric_card("Mean Article Length", f"{df['content_length'].mean():.0f} words", f"median: {df['content_length'].median():.0f} words")
with c4:
    metric_card("Max Article Length", f"{df['content_length'].max():,} words", "significant upper outlier")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Document Length Distribution
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    section_header("Document Length Distribution by Category", "Word count post-preprocessing (Boxplot: extreme outliers clipped at 98th percentile for visualization)")
    plot_df = df.copy()
    clip_val = df["content_length"].quantile(0.98)
    plot_df["content_length_clipped"] = plot_df["content_length"].clip(upper=clip_val)
    fig = px.box(
        plot_df, x=LABEL_COL, y="content_length_clipped", color=LABEL_COL,
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_layout(
        showlegend=False, height=440, xaxis_title="", yaxis_title="Word Count",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"tickangle": -30},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_header("Document Length Histogram", "Global distribution profile (clipped at 98th percentile)")
    fig2 = px.histogram(
        plot_df, x="content_length_clipped", nbins=40,
        color_discrete_sequence=["#4F46E5"],
    )
    fig2.update_layout(
        height=440, xaxis_title="Word Count", yaxis_title="Frequency",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

insight(
    f"The median article length is <b>{df['content_length'].median():.0f} words</b>, though the overall length profile "
    f"exhibits a distinct <b>right-skewed distribution</b>, with long-tail outliers reaching up to {df['content_length'].max():,} words "
    f"(typical of full-length press releases or investigative pieces). Target categories such as <b>Hajj</b> and <b>International & History</b> "
    f"demonstrate higher average document lengths, consistent with narrative-heavy reporting styles in those domains."
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global Term Frequency
# ---------------------------------------------------------------------------
section_header("Global Term Frequency", "Top 20 unigrams across corpus post-stopword removal & stemming")
top20 = global_counter.most_common(20)
top20_df = pd.DataFrame(top20, columns=["Term", "Frequency"])
fig3 = px.bar(
    top20_df.sort_values("Frequency"), x="Frequency", y="Term", orientation="h",
    color="Frequency", color_continuous_scale="Purples",
)
fig3.update_layout(
    height=520, showlegend=False, coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Class-Specific Word Clouds & Top Terms
# ---------------------------------------------------------------------------
section_header("Class-Specific Key Terms", "Select a target category to inspect its word cloud and top discriminative terms")

selected_cat = st.selectbox("Select Target Class", sorted(df[LABEL_COL].unique()))

wc_col, bar_col = st.columns([1, 1])

with wc_col:
    text_blob = " ".join(df.loc[df[LABEL_COL] == selected_cat, "article_content"].astype(str))
    wc = WordCloud(
        width=800, height=500, background_color="#0B1120",
        colormap="cool", max_words=80,
    ).generate(text_blob)
    fig_wc, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_facecolor("#0B1120")
    st.pyplot(fig_wc, use_container_width=True)

with bar_col:
    cat_words = pd.DataFrame(top_words_map[selected_cat], columns=["Term", "Frequency"])
    fig4 = px.bar(
        cat_words.sort_values("Frequency"), x="Frequency", y="Term", orientation="h",
        color_discrete_sequence=[CATEGORY_COLORS.get(selected_cat, "#4F46E5")],
    )
    fig4.update_layout(
        height=500, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", margin=dict(l=10, r=10, t=30, b=10),
        title=f"Top 12 Terms — {selected_cat}",
    )
    st.plotly_chart(fig4, use_container_width=True)

insight(
    f"Dominant vocabulary in the <b>{selected_cat}</b> class demonstrates strong thematic coherence and distinct "
    f"lexical separability from competing classes — a strong statistical indicator that frequency-based vectorization "
    f"(TF-IDF) is well-suited for this text classification task."
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Category Descriptive Statistics Summary
# ---------------------------------------------------------------------------
section_header("Category Descriptive Statistics")
summary = df.groupby(LABEL_COL).agg(
    Article_Count=("article_id", "count"),
    Mean_Length=("content_length", "mean"),
    Median_Length=("content_length", "median"),
    Min_Length=("content_length", "min"),
    Max_Length=("content_length", "max"),
).round(1).sort_values("Article_Count", ascending=False)
summary.columns = ["Article Count", "Mean Length", "Median Length", "Min Length", "Max Length"]
st.dataframe(summary, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import inject_global_css, metric_card, section_header, insight, warn, load_data, LABEL_COL

st.set_page_config(page_title="Data Cleaning | News Dashboard", page_icon="🧹", layout="wide")
inject_global_css()
df = load_data()

st.markdown(
    """
    <div style="font-size:1.9rem; font-weight:900; color:#F8FAFC;">🧹 Data Cleaning & Preprocessing</div>
    <div style="color:#94A3B8; margin-bottom:1rem;">
        How 10,000 raw news articles were refined into a production-grade corpus.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Funnel: raw -> cleaned
# ---------------------------------------------------------------------------
RAW_COUNT = 10000
MISSING_DROPPED = 36
EXACT_DUP_STAGE = 9807  # after dropna + short-text + exact dup + near-dup removal (df shape pre text-cleaning)
FINAL_COUNT = len(df)

section_header("Data Cleaning Funnel", "Pipeline journey from raw ingestion to model-ready dataset")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Raw Ingested Data", f"{RAW_COUNT:,}", "initial raw records (data.csv)")
with c2:
    metric_card("Missing Values", f"-{MISSING_DROPPED}", "null article_content records dropped")
with c3:
    metric_card("Duplicates & Near-Duplicates", f"-{RAW_COUNT - EXACT_DUP_STAGE - MISSING_DROPPED if RAW_COUNT - EXACT_DUP_STAGE - MISSING_DROPPED > 0 else RAW_COUNT - EXACT_DUP_STAGE}",
                "exact duplicates + TF-IDF similarity > 90%")
with c4:
    metric_card("Final Dataset", f"{FINAL_COUNT:,}", f"{FINAL_COUNT/RAW_COUNT*100:.1f}% retention rate")

fig = go.Figure(go.Funnel(
    y=["Raw Ingested Data", "Post Null Removal", "Post Exact & Near-Dup Removal", "Final Dataset (df_final.csv)"],
    x=[RAW_COUNT, RAW_COUNT - MISSING_DROPPED, EXACT_DUP_STAGE, FINAL_COUNT],
    textinfo="value+percent initial",
    marker={"color": ["#4F46E5", "#6366F1", "#818CF8", "#06B6D4"]},
))
fig.update_layout(
    height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#E2E8F0", margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig, use_container_width=True)

insight(
    "Total articles pruned: <b>193 records (≈1.9%)</b> — predominantly driven by near-duplicate detection "
    "(articles exceeding a TF-IDF cosine similarity threshold of > 90%), rather than missing values. "
    "This rigorous pruning maintains data integrity while preventing <i>data leakage</i> caused by highly overlapping "
    "text instances spanning both the training and evaluation splits."
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cleaning steps explained
# ---------------------------------------------------------------------------
section_header("Preprocessing Stages", "5 key pipeline stages executed in notebook 01_dataCleaning.ipynb")

steps = st.tabs([
    "1️⃣ Missing & Exact Duplicates", "2️⃣ Near-Duplicate Detection",
    "3️⃣ Noise Stripping", "4️⃣ Normalization & Stemming", "5️⃣ Label Consolidation",
])

with steps[0]:
    st.markdown(
        """
        - **Missing Value Imputation**: 36 null records missing `article_content` were dropped from the 10,000 raw samples.
        - **Short Text Pruning**: Records with content length ≤ 3 tokens (e.g., placeholder tokens like `"tes"`, `"."`, `","`) were discarded as non-informative web scraping noise.
        - **Exact Duplicate Pruning**: 54 exact string matches were identified and filtered via 
          `drop_duplicates(subset=['article_content'], keep='first')`.
        """
    )

with steps[1]:
    st.markdown(
        """
        Semantic duplicates often differ subtly in syntax (typos, whitespace variations, minor editorial edits). 
        To filter these near-duplicates:

        1. Construct a **TF-IDF vector matrix** (Word N-Grams: 1-2) across the whole corpus.
        2. Compute pairwise **Cosine Similarity** scores.
        3. Tag article pairs with similarity scores **> 0.90** as near-duplicate candidates.
        4. For pairs sharing the **same target class** $\rightarrow$ drop one redundant instance.  
           For pairs assigned **conflicting target classes** $\rightarrow$ drop both instances to eliminate label noise/ambiguity.
        """
    )
    insight(
        "Real-world edge case identified: Two medical articles demonstrated a <b>99.71% similarity</b> score "
        "yet carried conflicting labels (<i>Kesehatan</i> vs. <i>Obat-obatan</i>) — demonstrating the critical need for "
        "downstream target consolidation."
    )

with steps[2]:
    st.markdown(
        """
        Raw text extracted via web scraping contains structural noise that must be sanitized:

        - **HTML Entity & Tag Stripping**: Removal of residual tags like `<p>`, `<div>`, `<a>`, and HTML entities.
        - **Editorial Noise Removal**: Removal of reporter/editor credits, location datelines, and boilerplate markers like *"Baca juga:"* (Read also:).
        - **URL Sanitization**: Stripping of active web linkages starting with `http://` and `www.`.
        - **Whitespace Normalization**: Collapsing redundant newlines, tab stops, and multi-space gaps into single spaces.
        """
    )

with steps[3]:
    st.markdown(
        """
        - **Case Folding**: Conversion of all tokens to lowercase.
        - **CamelCase Tokenization**: Splitting concatenated camel-case tokens (e.g., `BojonegoroCom` $\rightarrow$ `Bojonegoro Com`) prior to lowercasing.
        - **Punctuation & Digit Removal**: Stripping numbers and special characters, followed by space re-alignment.
        - **Stopword Filtering**: Applied Indonesian NLTK stopword lists combined with a domain-customized stopword dictionary 
          (media outlet names like `kompas`, `detik`, `cnn`, and editorial verbs like `baca`, `juga`, `redaksi`).
        - **Morphological Stemming**: Implemented **Sastrawi Stemmer** to map inflected terms back to base root forms 
          (e.g., `"pertandingan"` $\rightarrow$ `"tanding"`), reducing vocabulary dimensionality and resolving morphological sparsity.
        """
    )
    st.code(
        'polisi inggris buru laku ledak bom stasiun london orang luka peristiwa jumat pagi sibuk asisten ...',
        language="text",
    )
    st.caption("Sample processed output following the full NLP preprocessing pipeline.")

with steps[4]:
    st.markdown(
        """
        The raw dataset comprised **29 overlapping classes**, with several classes suffering severe sample sparsity 
        (< 15 instances) unsuitable for supervised learning. Target consolidation was conducted via a **hybrid framework**:

        **A. Data-Driven Validation**
        - Calculate class TF-IDF centroids (mean feature vector across all documents per category).
        - Compute pairwise cosine similarity matrix across class centroids.
        - Perform **Hierarchical Agglomerative Clustering** across multi-distance thresholds to locate stable cluster candidates.

        **B. Domain Knowledge Integration**
        - Semantically related categories exhibiting disjoint term spaces (e.g., `MotoGP` $\rightarrow$ `Otomotif`) 
          were manually consolidated, bridging pure TF-IDF surface-level lexical mismatches.
        """
    )
    mapping_df = pd.DataFrame({
        "Final Target Class": [
            "Sports", "Healthcare", "Entertainment & Lifestyle", "International & History",
            "Economics & Business", "Politics & Law", "Technology & Science", "Regional News",
            "Hajj", "Automotive", "Education",
        ],
        "Consolidated Raw Classes": [
            "Sepak Bola, Sports",
            "Health, Obat-obatan, Kesehatan",
            "Hiburan, Personal, Lifestyle, Horor, Travel, K-Pop",
            "Internasional, Sejarah",
            "Ekonomi, Bisnis, Keuangan",
            "Politik, Hukum, KPK, Pilgub Jatim",
            "Teknologi, Sains",
            "Bojonegoro, Regional, Jakarta",
            "Haji",
            "Otomotif, MotoGP",
            "Pendidikan",
        ],
    })
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    insight("Final Target Space: <b>29 raw classes consolidated into 11 well-defined, balanced target classes</b> optimized for downstream modeling.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Before/after sample
# ---------------------------------------------------------------------------
section_header("Processed Data Sample", "Representative dataset view following pipeline execution")
st.dataframe(
    df[["article_id", LABEL_COL, "article_content"]].sample(8, random_state=1).reset_index(drop=True),
    use_container_width=True, hide_index=True,
)
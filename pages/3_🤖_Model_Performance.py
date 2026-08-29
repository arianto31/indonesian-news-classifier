import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import inject_global_css, metric_card, section_header, insight, warn, load_metrics, CATEGORY_COLORS

st.set_page_config(page_title="Model Performance | News Dashboard", page_icon="🤖", layout="wide")
inject_global_css()
metrics = load_metrics()

st.markdown(
    """
    <div style="font-size:1.9rem; font-weight:900; color:#F8FAFC;">🤖 Model Performance</div>
    <div style="color:#94A3B8; margin-bottom:1rem;">
        Experimental trajectory from baseline algorithms to the final production model — all metrics derived 
        from evaluation on a stratified held-out test set (20%, unexposed during training).
    </div>
    """,
    unsafe_allow_html=True,
)

report = metrics["classification_report"]
labels_order = metrics["labels_order"]
cm = np.array(metrics["confusion_matrix"])

accuracy = report["accuracy"]
macro_f1 = report["macro avg"]["f1-score"]
weighted_f1 = report["weighted avg"]["f1-score"]
macro_precision = report["macro avg"]["precision"]
macro_recall = report["macro avg"]["recall"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Accuracy", f"{accuracy*100:.2f}%", f"test set size n = {metrics['test_size']}")
with c2:
    metric_card("Macro F1-Score", f"{macro_f1:.4f}", "unweighted average across classes")
with c3:
    metric_card("Macro Precision", f"{macro_precision:.4f}", "mean positive predictive value")
with c4:
    metric_card("Macro Recall", f"{macro_recall:.4f}", "mean sensitivity / true positive rate")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Experiment comparison (from notebook 03_model.ipynb cross-validation runs)
# ---------------------------------------------------------------------------
section_header("Model Experimentation Benchmarks", "5-fold cross-validation Macro F1-Scores evaluated on the training set")

algo_compare = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "Linear SVM", "SGD Classifier"],
    "Macro F1 (CV Mean)": [0.8810, 0.7536, 0.8953, 0.8869],
    "Std Dev": [0.0054, 0.0070, 0.0125, 0.0055],
})
fig_algo = px.bar(
    algo_compare.sort_values("Macro F1 (CV Mean)"), x="Macro F1 (CV Mean)", y="Model", orientation="h",
    error_x="Std Dev", color="Macro F1 (CV Mean)", color_continuous_scale="Viridis", text="Macro F1 (CV Mean)",
)
fig_algo.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig_algo.update_layout(
    height=320, coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_algo, use_container_width=True)
insight(
    "Baseline benchmark across <b>4 candidate algorithms</b> evaluated via word N-gram (1,2) TF-IDF features + 5-fold cross-validation: "
    "<b>Naive Bayes</b> severely underperformed (0.754 Macro F1) due to strong feature independence assumptions that fail on longer text documents, "
    "whereas <b>Linear SVM</b> yielded superior performance (0.895 Macro F1) given its effectiveness in high-dimensional, sparse feature spaces."
)

st.markdown("<br>", unsafe_allow_html=True)

section_header("Feature Engineering & Hyperparameter Optimization", "Iterative enhancements evaluated on top of the baseline Linear SVM model")

exp_df = pd.DataFrame({
    "Experiment": ["Word + Char TF-IDF", "Baseline (word only)", "Tuned C (grid search)", "Ensemble (SVM+LogReg+SGD)"],
    "Macro F1 Mean": [0.896554, 0.895301, 0.894882, 0.891666],
    "Std Dev": [0.008800, 0.012521, 0.007645, 0.007833],
})
fig_exp = px.bar(
    exp_df.sort_values("Macro F1 Mean"), x="Macro F1 Mean", y="Experiment", orientation="h",
    error_x="Std Dev", color="Macro F1 Mean", color_continuous_scale="Blues", text="Macro F1 Mean",
)
fig_exp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig_exp.update_layout(
    height=340, coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
    xaxis_range=[0.885, 0.90],
)
st.plotly_chart(fig_exp, use_container_width=True)

insight(
    "Incorporating sub-word <b>character N-grams (3-5 char boundary range)</b> alongside word N-grams provided a slight "
    "yet consistent improvement (+0.0013 Macro F1), as character-level features effectively capture morphological variants and typos "
    "bypassed by stemming. Interestingly, an <b>Ensemble Voting Classifier</b> (SVM + Logistic Regression + SGD) resulted in a marginal performance drop "
    "(0.8917 Macro F1) — indicating correlated misclassification errors on minority classes where voting offered no variance reduction. "
    "The final production architecture selected is a <b>Linear SVM + Word & Char TF-IDF + Tuned $C = 0.3149$</b>, balancing classification accuracy "
    "with low computational complexity."
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Classification report (final model)
# ---------------------------------------------------------------------------
section_header("Classification Diagnostics — Final Model", "Per-class Precision, Recall, and F1-Score breakdown on the held-out test set")

rows = []
for label in labels_order:
    r = report[label]
    rows.append({
        "Category": label,
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-Score": r["f1-score"],
        "Support": int(r["support"]),
    })
report_df = pd.DataFrame(rows).sort_values("Support", ascending=False)

left, right = st.columns([1.3, 1])
with left:
    st.dataframe(
        report_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"})
        .background_gradient(subset=["F1-Score"], cmap="RdYlGn", vmin=0.7, vmax=1.0),
        use_container_width=True, hide_index=True, height=430,
    )
with right:
    fig_f1 = px.bar(
        report_df.sort_values("F1-Score"), x="F1-Score", y="Category", orientation="h",
        color="F1-Score", color_continuous_scale="RdYlGn", range_color=[0.7, 1.0],
    )
    fig_f1.update_layout(
        height=430, coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

weakest = report_df.sort_values("F1-Score").iloc[0]
warn(
    f"The <b>{weakest['Category']}</b> category recorded the lowest performance ({weakest['F1-Score']:.2f} F1-score) with a support of "
    f"only {int(weakest['Support'])} instances in the test set — representing the most underrepresented class in the corpus. "
    f"This represents a classic imbalanced learning tradeoff: acquiring additional training instances for rare classes is expected "
    f"to directly improve decision boundary quality."
)
insight(
    "Domains such as <b>Hajj</b> and <b>Sports</b> achieved high F1-scores (≈0.98–0.99). The domain vocabulary for these topics "
    "is highly distinctive and orthogonal to other classes, minimizing cross-class semantic confusion."
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
section_header("Confusion Matrix Heatmap", "Rows = Ground Truth Classes, Columns = Predicted Classes")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig_cm = go.Figure(data=go.Heatmap(
    z=cm_norm, x=labels_order, y=labels_order,
    colorscale="Blues", text=cm, texttemplate="%{text}",
    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
))
fig_cm.update_layout(
    height=620, xaxis_title="Predicted Class", yaxis_title="Ground Truth Class",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#E2E8F0", margin=dict(l=10, r=10, t=10, b=10),
    yaxis={"autorange": "reversed"},
)
st.plotly_chart(fig_cm, use_container_width=True)

# find biggest off-diagonal confusion
cm_off = cm.copy().astype(float)
np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(np.argmax(cm_off), cm_off.shape)
insight(
    f"The primary off-diagonal misclassification occurred when instances of <b>{labels_order[i]}</b> were mispredicted as "
    f"<b>{labels_order[j]}</b> ({int(cm_off[i,j])} misclassifications). These target classes share overlapping vocabulary spaces "
    f"(e.g., policy/economic news intersecting with legislative affairs or international historical analysis)."
)

st.markdown("<br>", unsafe_allow_html=True)
section_header("Final Pipeline Specification")
st.code(
    """
Pipeline Architecture:
  1. FeatureUnion
     - word_vectorizer: TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2, max_df=0.9, sublinear_tf=True)
     - char_vectorizer: TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=3000, min_df=2, sublinear_tf=True)
  2. Classifier:
     - LinearSVC(C=0.3149, class_weight='balanced', max_iter=3000, random_state=42)
""",
    language="yaml",
)
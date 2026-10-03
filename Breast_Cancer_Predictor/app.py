"""Interactive prediction dashboard (Streamlit).  Run with:  streamlit run app.py"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import model_core as mc

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

GREEN, RED, GREY = "#2e9e5b", "#d64545", "#b9bfc7"


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return mc.load_artifacts()


try:
    model, meta, source = get_artifacts()
except Exception as exc:                      # graceful failure instead of a stack trace
    st.error("The model could not be loaded or trained. Please check that the `data/` folder is present.")
    st.caption(f"Technical details: {exc}")
    st.stop()

RANGES = meta["ranges"]


# ------------------------------------------------------------------ helpers
def set_values(values):
    for f in mc.FEATURES:
        st.session_state[f"in_{f}"] = float(round(values[f], 4))


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def prob_chart(p_malignant):
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    labels, probs, colors = ["Benign", "Malignant"], [1 - p_malignant, p_malignant], [GREEN, RED]
    bars = ax.barh(labels, probs, color=colors)
    for b, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.9), b.get_y() + b.get_height() / 2, f"{p:.1%}", va="center", fontweight="bold")
    ax.set_xlim(0, 1.05); ax.set_xlabel("Predicted probability")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def meter_chart(p_malignant):
    fig, ax = plt.subplots(figsize=(5.5, 1.1))
    ax.barh([0], [1], color=GREY, height=0.5)
    ax.barh([0], [p_malignant], color=RED if p_malignant >= 0.5 else GREEN, height=0.5)
    ax.axvline(0.5, color="black", lw=1.5, ls="--")
    ax.text(0.5, 0.42, "decision threshold (50%)", ha="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Malignancy risk meter", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def contribution_chart(contrib, kind):
    s = contrib.sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    colors = [RED if v > 0 else GREEN for v in s.values] if kind == "local" else "steelblue"
    ax.barh([mc.pretty(i) for i in s.index], s.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Push towards malignant (+) / benign (−)" if kind == "local" else "Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ header
st.title("🩺 Breast Cancer Prediction Dashboard")
st.caption("Machine-learning model trained on the Wisconsin Diagnostic Breast Cancer dataset: it predicts whether a "
           "tumour is **benign** or **malignant** from cell-nucleus measurements.")
st.warning("**Educational demo only.** This is not a medical device and must not be used for diagnosis or "
           "treatment decisions. Always consult a qualified doctor.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Open **Single prediction** and enter the 10 measurements, or click a **sample** button.\n"
        "2. The result updates **instantly** as you change a value.\n"
        "3. Open **Batch (CSV)** to predict many cases at once.\n"
        "4. See **Model performance** for metrics and charts.")
    st.divider()
    st.subheader("Model")
    st.write(f"**{meta['model_name']}**")
    tm = meta["test_metrics"]
    st.write(f"Test accuracy: **{tm['accuracy']:.1%}**  \nTest ROC-AUC: **{tm['roc_auc']:.3f}**")
    st.caption(f"Status: {source}")

tab_single, tab_batch, tab_perf, tab_about = st.tabs(
    ["🧍 Single prediction", "📁 Batch (CSV)", "📊 Model performance", "ℹ️ About"])

# ------------------------------------------------------------------ tab 1: single prediction
with tab_single:
    for f in mc.FEATURES:                                       # defaults = typical benign case
        st.session_state.setdefault(f"in_{f}", float(round(meta["class_medians"]["benign"][f], 4)))

    b1, b2, b3, _ = st.columns([1.3, 1.5, 1, 3])
    b1.button("Load benign sample", on_click=set_values, args=(meta["class_medians"]["benign"],))
    b2.button("Load malignant sample", on_click=set_values, args=(meta["class_medians"]["malignant"],))
    b3.button("Reset", on_click=set_values, args=(meta["class_medians"]["benign"],))

    left, right = st.columns([1, 1.15], gap="large")
    values = {}
    with left:
        st.subheader("Enter measurements")
        c1, c2 = st.columns(2)
        for i, f in enumerate(mc.FEATURES):
            r = RANGES[f]
            with (c1 if i % 2 == 0 else c2):
                values[f] = st.number_input(
                    mc.pretty(f), min_value=0.0, step=max(round((r["max"] - r["min"]) / 100, 4), 0.0001),
                    format="%.4f", key=f"in_{f}",
                    help=f"{mc.FEATURE_INFO[f]}. Training range: {r['min']:.4g} to {r['max']:.4g}.")

    with right:
        st.subheader("Prediction")
        errors, warnings = mc.validate_values(values, meta)
        if errors:
            for e in errors:
                st.error(e)
            st.info("Fix the highlighted problems to see a prediction.")
        else:
            try:
                p = float(mc.predict_proba_malignant(model, [list(values.values())])[0])
            except Exception as exc:
                st.error("Prediction failed. Please check your inputs and try again.")
                st.caption(f"Technical details: {exc}")
                st.stop()
            for w in warnings:
                st.warning(w)
            label = "Malignant" if p >= 0.5 else "Benign"
            confidence = max(p, 1 - p)
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", label)
            m2.metric("Confidence", f"{confidence:.1%}")
            m3.metric("P(malignant)", f"{p:.1%}")
            (st.warning if label == "Malignant" else st.success)(
                f"The model predicts **{label.lower()}** with {confidence:.1%} confidence.")
            show_fig(meter_chart(p))
            show_fig(prob_chart(p))

    if not errors:
        contrib, kind = mc.explain(model, values)
        if contrib is not None:
            st.subheader("Why this prediction?" if kind == "local" else "Which features matter most?")
            st.caption("Bars to the right (red) push the model towards *malignant*; bars to the left (green) "
                       "push towards *benign*." if kind == "local" else "Global feature importance of the model.")
            show_fig(contribution_chart(contrib, kind))

# ------------------------------------------------------------------ tab 2: batch
with tab_batch:
    st.subheader("Predict many cases from a CSV file")
    st.markdown("Upload a **.csv** file with these columns (any order, extra columns are ignored):")
    st.code(", ".join(mc.FEATURES))
    template = pd.DataFrame([meta["class_medians"]["benign"], meta["class_medians"]["malignant"]])[mc.FEATURES].round(4)
    st.download_button("⬇️ Download a template CSV", template.to_csv(index=False).encode(),
                       "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()))
        except pd.errors.EmptyDataError:
            st.error("The file is empty.")
            raw = None
        except Exception:
            st.error("Could not read this file. Please upload a valid, comma-separated .csv file.")
            raw = None
        if raw is not None:
            valid, problems = mc.prepare_batch(raw, meta)
            for msg in problems:
                (st.error if valid is None else st.warning)(msg)
            if valid is not None:
                p = mc.predict_proba_malignant(model, valid.values)
                out = valid.copy()
                out["p_malignant"] = p.round(4)
                out["prediction"] = np.where(p >= 0.5, "Malignant", "Benign")
                n_m = int((p >= 0.5).sum())
                s1, s2, s3 = st.columns(3)
                s1.metric("Rows predicted", len(out)); s2.metric("Benign", len(out) - n_m); s3.metric("Malignant", n_m)
                st.dataframe(out, width="stretch")
                st.download_button("⬇️ Download predictions", out.to_csv().encode(), "predictions.csv", "text/csv")
                fig, ax = plt.subplots(figsize=(4.5, 2.5))
                ax.bar(["Benign", "Malignant"], [len(out) - n_m, n_m], color=[GREEN, RED])
                ax.set_ylabel("Cases"); ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout(); show_fig(fig)

# ------------------------------------------------------------------ tab 3: performance
with tab_perf:
    st.subheader("Held-out test set results")
    st.caption(f"Model: **{meta['model_name']}** · trained on {meta['n_train']} cases, "
               f"tested on {meta['n_test']} unseen cases.")
    tm = meta["test_metrics"]
    cols = st.columns(5)
    for c, (k, name) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                   ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]):
        c.metric(name, f"{tm[k]:.3f}")

    g1, g2 = st.columns(2)
    with g1:
        cm = np.array(meta["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Benign", "Malignant"]); ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        fig.tight_layout(); show_fig(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.plot(meta["roc"]["fpr"], meta["roc"]["tpr"], color="steelblue", lw=2, label=f"AUC = {tm['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve")
        ax.legend(loc="lower right"); fig.tight_layout(); show_fig(fig)

    st.subheader("Model comparison (5-fold cross-validation on the training set)")
    comp = pd.DataFrame(meta["comparison"]).T.rename(
        columns={"cv_roc_auc": "CV ROC-AUC", "cv_accuracy": "CV accuracy"}).round(4)
    st.dataframe(comp, width="stretch")
    st.caption("The three models score almost the same, so the simpler and explainable Logistic Regression is "
               "preferred when it is within 0.005 ROC-AUC of the best model.")

# ------------------------------------------------------------------ tab 4: about
with tab_about:
    st.subheader("About this project")
    st.markdown(
        "- **Task:** binary classification (benign vs malignant).\n"
        "- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository "
        "(569 cases, 10 mean cell-nucleus features used here).\n"
        "- **Pipeline:** stratified train/test split → StandardScaler + classifier inside one scikit-learn "
        "`Pipeline` (no data leakage) → model saved with `joblib`.\n"
        "- **Validation:** missing, non-positive, implausibly large or inconsistent values are flagged; "
        "values outside the training range trigger a reliability warning.\n"
        "- **Limits:** only 569 cases and 10 features, so the model is a learning demo, not a clinical tool. "
        f"About {1 - tm['accuracy']:.0%} of test cases were still misclassified, including missed malignant cases.")
    st.markdown("**Feature meanings**")
    st.dataframe(pd.DataFrame({"Feature": [mc.pretty(f) for f in mc.FEATURES],
                               "Meaning": [mc.FEATURE_INFO[f] for f in mc.FEATURES],
                               "Training range": [f"{RANGES[f]['min']:.4g} – {RANGES[f]['max']:.4g}"
                                                  for f in mc.FEATURES]}),
                 width="stretch", hide_index=True)

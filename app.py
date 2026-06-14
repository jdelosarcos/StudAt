from pathlib import Path
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from preprocessing import clean_student_data, get_modeling_data, TARGET
from train_models import train_all

st.set_page_config(page_title="Student Degree Outcome Prediction", layout="wide")

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
DEFAULT_DATA = DATA_DIR / "StudProfile.xlsx"

st.title("Student Degree Outcome Prediction and Pattern Mining Dashboard")
st.caption("Educational data mining dashboard for identifying features influencing student completion, stopping, and shifting behavior.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select module", [
        "1. Upload and Overview",
        "2. Feature Relationships",
        "3. Train / Load Model",
        "4. Prediction and Model Results",
        "5. Feature Influence",
        "6. Association Rules",
        "7. Research Interpretation",
    ])
    st.divider()
    feature_set = st.selectbox("Model feature set", ["academic", "enrollment"], index=0,
                               help="Enrollment uses entry/profile features only. Academic includes first-year grades.")

@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def clean_cached(raw_df):
    return clean_student_data(raw_df)


def load_raw_data():
    uploaded = st.sidebar.file_uploader("Upload Excel dataset", type=["xlsx", "xls"])
    if uploaded is not None:
        raw = read_excel_cached(uploaded.getvalue())
        return raw, uploaded.name
    if DEFAULT_DATA.exists():
        raw = pd.read_excel(DEFAULT_DATA)
        return raw, str(DEFAULT_DATA)
    return None, None


def metric_cards(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Completed", int((df[TARGET] == "Completed").sum()))
    c4.metric("Stopped", int((df[TARGET] == "Stopped").sum()))


def model_files_exist():
    return (MODELS_DIR / "outcome_model.pkl").exists() and (MODELS_DIR / "feature_list.pkl").exists()


def load_model_package():
    model = joblib.load(MODELS_DIR / "outcome_model.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    metadata = {}
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    return model, features, metadata


def create_excel_download(dfs: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

raw_df, source_name = load_raw_data()
if raw_df is None:
    st.info("Upload your Excel file from the sidebar to begin. You may also place it in `data/StudProfile.xlsx` for default loading.")
    st.stop()

clean_df = clean_cached(raw_df)
modeling_classes = [c for c, n in clean_df[TARGET].value_counts().items() if c in ["Completed", "Stopped", "Shifted"] and n >= 2]

if page == "1. Upload and Overview":
    st.subheader("Dataset Overview")
    st.write(f"Loaded source: `{source_name}`")
    metric_cards(clean_df)

    st.markdown("### Degree behavior distribution")
    status_counts = clean_df[TARGET].value_counts().reset_index()
    status_counts.columns = ["Degree Status", "Count"]
    fig = px.bar(status_counts, x="Degree Status", y="Count", text="Count", title="Degree Status Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.info("This chart shows whether the dataset is imbalanced. If one class dominates, accuracy alone can be misleading; F1-macro and balanced accuracy should be prioritized.")

    st.markdown("### Missing values")
    missing = (clean_df.isna().mean() * 100).sort_values(ascending=False).head(20).reset_index()
    missing.columns = ["Variable", "Missing Percent"]
    fig2 = px.bar(missing, x="Missing Percent", y="Variable", orientation="h", title="Top 20 Variables with Missing Values")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Cleaned data preview")
    st.dataframe(clean_df.head(30), use_container_width=True)

elif page == "2. Feature Relationships":
    st.subheader("Feature Relationships with Degree Outcome")
    model3_df = clean_df[clean_df[TARGET].isin(["Completed", "Stopped", "Shifted"])].copy()
    feature = st.selectbox("Select categorical feature", [
        "course", "shs_strand", "has_scholarship", "has_extracurricular", "family_income", "residence", "mother_occ_group", "father_occ_group", "performance_1st_year"
    ])
    temp = pd.crosstab(model3_df[feature], model3_df[TARGET], normalize="index").mul(100).reset_index()
    long = temp.melt(id_vars=feature, var_name="Degree Status", value_name="Percentage")
    fig = px.bar(long, x=feature, y="Percentage", color="Degree Status", title=f"Degree Status by {feature}")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Interpretation: Higher percentages for `Stopped` or `Shifted` under a category indicate possible risk patterns that should be further examined.")
    st.dataframe(temp.round(2), use_container_width=True)

    numeric = st.selectbox("Select numeric feature", ["age_first_year", "hs_average", "income_rank", "intro_computing", "programming", "gwa_1st_year", "extracurricular_count"])
    fig2 = px.box(model3_df, x=TARGET, y=numeric, points="all", title=f"{numeric} by Degree Status")
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Interpretation: Heavy overlap among boxplots suggests that the classes are naturally difficult to separate, which explains lower predictive accuracy.")

elif page == "3. Train / Load Model":
    st.subheader("Train or Load Prediction Model")
    st.write("This deployment uses stable models suitable for free cloud deployment: balanced logistic regression, balanced random forest, extra trees, balanced random forest, and easy ensemble.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train model using current uploaded dataset", type="primary"):
            temp_path = DATA_DIR / "uploaded_training_data.xlsx"
            raw_df.to_excel(temp_path, index=False)
            with st.spinner("Training models. Please wait..."):
                results, metadata = train_all(temp_path, feature_set=feature_set)
            st.success("Training completed and model files were saved in the models folder.")
            st.json(metadata)
            st.dataframe(results.head(15), use_container_width=True)
    with c2:
        if model_files_exist():
            model, features, metadata = load_model_package()
            st.success("Saved model files found.")
            st.json(metadata)
        else:
            st.warning("No saved model found yet. Click the training button first.")

elif page == "4. Prediction and Model Results":
    st.subheader("Prediction and Model Results")
    if not model_files_exist():
        st.warning("No model files found. Go to `Train / Load Model` first.")
        st.stop()
    model, features, metadata = load_model_package()
    available = [f for f in features if f in clean_df.columns]
    X = clean_df[available].copy()
    preds = model.predict(X)
    result_df = clean_df.copy()
    result_df["Predicted_Degree_Status"] = preds
    if hasattr(model.named_steps.get("model"), "predict_proba"):
        try:
            proba = model.predict_proba(X)
            result_df["Prediction_Confidence"] = proba.max(axis=1)
        except Exception:
            pass

    st.markdown("### Prediction distribution")
    pred_counts = result_df["Predicted_Degree_Status"].value_counts().reset_index()
    pred_counts.columns = ["Predicted Status", "Count"]
    fig = px.pie(pred_counts, names="Predicted Status", values="Count", title="Predicted Degree Outcome Distribution")
    st.plotly_chart(fig, use_container_width=True)

    valid = result_df[TARGET].isin(metadata.get("classes", []))
    if valid.sum() > 0:
        y_true = result_df.loc[valid, TARGET]
        y_pred = result_df.loc[valid, "Predicted_Degree_Status"]
        m = pd.DataFrame([{
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }])
        st.markdown("### Metrics on current dataset")
        st.dataframe(m, use_container_width=True)
        cm = confusion_matrix(y_true, y_pred, labels=metadata.get("classes", []))
        cm_df = pd.DataFrame(cm, index=metadata.get("classes", []), columns=metadata.get("classes", []))
        fig2 = px.imshow(cm_df, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Prediction table")
    st.dataframe(result_df.head(100), use_container_width=True)
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv, file_name="student_degree_predictions.csv", mime="text/csv")

elif page == "5. Feature Influence":
    st.subheader("Feature Influence Analysis")
    if not model_files_exist():
        st.warning("Train or load a model first.")
        st.stop()
    model, features, metadata = load_model_package()
    X, y, features2, modeling_df, classes = get_modeling_data(clean_df, feature_set=metadata.get("feature_set", feature_set), classes=metadata.get("classes"))
    st.write("Permutation importance identifies which variables reduce F1-macro most when shuffled.")
    if st.button("Compute feature influence"):
        with st.spinner("Computing permutation importance..."):
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="f1_macro", n_jobs=1)
            imp = pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
        st.dataframe(imp, use_container_width=True)
        fig = px.bar(imp.head(15).iloc[::-1], x="Importance", y="Feature", orientation="h", title="Top Feature Influence")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Variables at the top are the strongest available predictors. They should be interpreted as exploratory indicators, not automatic causal factors.")

elif page == "6. Association Rules":
    st.subheader("Association Rule Mining")
    df = clean_df[clean_df[TARGET].isin(["Completed", "Stopped", "Shifted"])].copy()
    min_support = st.slider("Minimum support", 0.01, 0.20, 0.04, 0.01)
    min_conf = st.slider("Minimum confidence", 0.30, 0.95, 0.50, 0.05)

    def add_bands(d):
        d = d.copy()
        d["age_band"] = pd.cut(d["age_first_year"], bins=[0,18,21,99], labels=["Age <=18", "Age 19-21", "Age >=22"])
        d["hs_average_band"] = pd.cut(d["hs_average"], bins=[0,79,89,100], labels=["HS low", "HS average", "HS high"])
        d["income_band"] = d["income_rank"].map({1:"Income very low",2:"Income low",3:"Income moderate",4:"Income upper moderate",5:"Income high",6:"Income very high"})
        return d

    if st.button("Generate association rules"):
        rules_df = add_bands(df)
        cols = ["degree_status", "course", "shs_strand", "has_scholarship", "has_extracurricular", "family_income", "residence", "mother_occ_group", "father_occ_group", "performance_1st_year", "programming_performance", "intro_performance", "age_band", "hs_average_band", "income_band"]
        cols = [c for c in cols if c in rules_df.columns]
        transactions = []
        for _, row in rules_df[cols].iterrows():
            items = [f"{col}={row[col]}" for col in cols if pd.notna(row[col]) and str(row[col]) != "Unknown"]
            transactions.append(items)
        te = TransactionEncoder()
        basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
        itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
        if len(itemsets) == 0:
            st.warning("No frequent itemsets found. Lower the minimum support.")
            st.stop()
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
        status_items = {f"degree_status={c}" for c in ["Completed", "Stopped", "Shifted"]}
        rules = rules[rules["consequents"].apply(lambda x: len(set(x) & status_items) > 0)].copy()
        if len(rules) == 0:
            st.warning("No status-related rules found. Lower support or confidence.")
            st.stop()
        rules["Antecedents"] = rules["antecedents"].apply(lambda x: " AND ".join(sorted(list(x))))
        rules["Consequents"] = rules["consequents"].apply(lambda x: " AND ".join(sorted(list(x))))
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
        show = rules[["Antecedents", "Consequents", "support", "confidence", "lift"]].head(30)
        st.dataframe(show, use_container_width=True)
        fig = px.bar(show.head(12).iloc[::-1], x="lift", y="Antecedents", color="Consequents", orientation="h", title="Top Association Rules by Lift")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Lift above 1 means the pattern is more common than expected by chance. Higher confidence means the consequent often follows the antecedent.")

elif page == "7. Research Interpretation":
    st.subheader("Research Interpretation Guide")
    dist = clean_df[TARGET].value_counts().to_dict()
    st.markdown(f"""
    ### Dataset Condition
    The degree behavior target distribution is `{dist}`. If the distribution is uneven, the dataset is imbalanced. Therefore, F1-macro and balanced accuracy are more appropriate than accuracy alone.

    ### Objective 1: Features influencing stopping, shifting, and completion
    Use the **Feature Relationships** and **Feature Influence** modules to identify variables associated with student outcomes. These outputs may be reported as exploratory predictors.

    ### Objective 2: Predictive model development
    Use the **Train / Load Model** module to build imbalance-aware models using either enrollment-only or academic-inclusive features.

    ### Objective 3: Patterns and relationships
    Use the **Association Rules** module to discover interpretable patterns such as combinations of strand, scholarship, academic performance, or income level linked to completion or stopping.

    ### Recommended Chapter 4 Statement
    The enhanced framework applied missing-value handling, rare-category grouping, imbalance-aware learning, cross-validation, feature influence analysis, and association rule mining. However, model performance should be interpreted cautiously because the dataset is small and the classes may overlap. The dashboard is best positioned as an exploratory early-warning and student profiling decision-support tool.
    """)

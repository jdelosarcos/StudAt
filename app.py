import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
APP_DIR = Path(__file__).parent
DEFAULT_DATA = APP_DIR / "data" / "StudProfile.xlsx"

st.set_page_config(
    page_title="Student Degree Outcome Prediction",
    page_icon="🎓",
    layout="wide",
)

# -----------------------------
# Utility and preprocessing
# -----------------------------

def clean_colname(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def midpoint_from_range(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    nums = re.findall(r"\d+\.?\d*", text)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def normalize_text(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "n/a", "na", "no answer", "null"]:
        return "Unknown"
    return re.sub(r"\s+", " ", s)


def extract_year(x):
    if pd.isna(x):
        return np.nan
    m = re.findall(r"(?:19|20)\d{2}", str(x))
    return float(m[0]) if m else np.nan


def derive_degree_status(x):
    if pd.isna(x):
        return "Other/Ongoing/Unknown"
    s = str(x).strip().lower()
    if any(k in s for k in ["shift", "shif"]):
        return "Shifted"
    if any(k in s for k in ["stop", "drop", "withdraw", "quit"]):
        return "Stopped"
    if any(k in s for k in ["enrolled", "ongoing", "currently", "continuing"]):
        return "Other/Ongoing/Unknown"
    if "graduate" in s:
        return "Completed"
    if not pd.isna(extract_year(s)):
        return "Completed"
    return "Other/Ongoing/Unknown"


def categorize_gwa_ph(value):
    if pd.isna(value):
        return "Unknown"
    if value <= 1.75:
        return "High performer"
    if value <= 2.50:
        return "Average performer"
    return "At-risk performer"


def income_rank(x):
    s = normalize_text(x).lower().replace(",", "")
    if "below 10000" in s or "10000 and below" in s or "10 000 and below" in s:
        return 1
    if "above 10000" in s and "below 20000" in s:
        return 2
    if "20000" in s and "30000" in s:
        return 3
    if "30000" in s and "40000" in s:
        return 4
    if "40000" in s and "50000" in s:
        return 5
    if "above 50000" in s or "50000 above" in s:
        return 6
    return np.nan


def has_scholarship(x):
    s = normalize_text(x).lower()
    if s in ["unknown", "none", "no", "n/a", "na"] or "no scholar" in s:
        return "No"
    return "Yes"


def extracurricular_count(x):
    s = normalize_text(x).lower()
    if s in ["unknown", "none", "no", "no activities", "no activity"] or "no activ" in s:
        return 0
    return max(1, len([p for p in re.split(r",|;|/", s) if p.strip()]))


def group_occupation(x):
    s = normalize_text(x).lower()
    if s == "unknown":
        return "Unknown"
    if any(k in s for k in ["farmer", "farm", "laborer", "labour"]):
        return "Farming/Labor"
    if any(k in s for k in ["teacher", "government", "employee", "office", "clerk", "nia", "deped", "barangay"]):
        return "Employed/Government"
    if any(k in s for k in ["business", "vendor", "store", "self", "entrepreneur"]):
        return "Business/Self-employed"
    if any(k in s for k in ["house", "wife", "home"]):
        return "Homemaker"
    if any(k in s for k in ["ofw", "abroad", "seaman"]):
        return "Overseas/Seafarer"
    return "Other"


def rare_group(series, min_count=4):
    s = series.astype(str).fillna("Unknown")
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), "Other/Rare")


@st.cache_data(show_spinner=False)
def load_excel(file_bytes_or_path):
    return pd.read_excel(file_bytes_or_path)


@st.cache_data(show_spinner=False)
def prepare_data(raw_df):
    clean_df = raw_df.copy()
    clean_df.columns = [clean_colname(c) for c in clean_df.columns]
    rename_map = {
        "unnamed_0": "id",
        "name_for_female_if_married_already_use_your_maiden_name": "name",
        "age_when_you_are_in_first_year_college": "age_first_year",
        "name_address_of_school_attended_in_high_school": "high_school",
        "general_average_obtained_in_high_school": "hs_average_raw",
        "civil_status_when_entered_in_college_asscat": "civil_status",
        "while_studying_in_asscat_are_you_staying_in": "residence",
        "extra_curricular_activities_involvement_check_all_that_applies": "extracurricular",
        "mother_occupation_when_you_are_still_in_college": "mother_occupation",
        "father_occupation_when_you_are_still_in_college": "father_occupation",
        "family_monthly_gross_income_when_you_are_still_in_college": "family_income",
        "strand_enrolled_in_senior_high": "shs_strand",
        "scholarship_availed_in_college_just_write_none_if_no_scholarship_availed": "scholarship",
        "grade_obtained_in_intro_to_computer_fundamentals_operations_or_introduction_to_computing": "intro_computing_raw",
        "grade_obtained_in_fundamentals_of_programming": "programming_raw",
        "general_grade_average_obtained_in_1st_year_college": "gwa_1st_year_raw",
        "year_started_in_college_asscat": "year_started",
        "year_graduated_in_college_asscat_if_stopped_or_shifted_to_other_course_pls_indicate_and_also_the_year_ex_stopped_2023": "year_graduated_raw",
        "ethnicity_ex_manobo_kamayo_etc_or_surigaonon_agusanon_bol_anon_cebuano_etc": "ethnicity",
        "course": "course",
    }
    clean_df = clean_df.rename(columns=rename_map)

    for col in ["hs_average_raw", "intro_computing_raw", "programming_raw", "gwa_1st_year_raw"]:
        if col in clean_df.columns:
            clean_df[col.replace("_raw", "")] = clean_df[col].apply(midpoint_from_range)

    for required in ["year_graduated_raw", "scholarship", "extracurricular", "family_income", "mother_occupation", "father_occupation"]:
        if required not in clean_df.columns:
            clean_df[required] = np.nan

    clean_df["age_first_year"] = pd.to_numeric(clean_df.get("age_first_year"), errors="coerce")
    clean_df["year_started"] = pd.to_numeric(clean_df.get("year_started"), errors="coerce")
    clean_df["year_graduated_numeric"] = clean_df["year_graduated_raw"].apply(extract_year)
    clean_df["degree_status"] = clean_df["year_graduated_raw"].apply(derive_degree_status)
    clean_df["time_to_completion"] = clean_df["year_graduated_numeric"] - clean_df["year_started"]
    clean_df["on_time_completion"] = np.where(
        (clean_df["degree_status"] == "Completed") & (clean_df["time_to_completion"] <= 4),
        "On-time",
        np.where(clean_df["degree_status"] == "Completed", "Delayed", "Not completed"),
    )
    clean_df["has_scholarship"] = clean_df["scholarship"].apply(has_scholarship)
    clean_df["extracurricular_count"] = clean_df["extracurricular"].apply(extracurricular_count)
    clean_df["has_extracurricular"] = np.where(clean_df["extracurricular_count"] > 0, "Yes", "No")
    clean_df["income_rank"] = clean_df["family_income"].apply(income_rank)
    clean_df["mother_occ_group"] = clean_df["mother_occupation"].apply(group_occupation)
    clean_df["father_occ_group"] = clean_df["father_occupation"].apply(group_occupation)

    for g in ["hs_average", "intro_computing", "programming", "gwa_1st_year"]:
        if g not in clean_df.columns:
            clean_df[g] = np.nan

    clean_df["performance_1st_year"] = clean_df["gwa_1st_year"].apply(categorize_gwa_ph)
    clean_df["programming_performance"] = clean_df["programming"].apply(categorize_gwa_ph)
    clean_df["intro_performance"] = clean_df["intro_computing"].apply(categorize_gwa_ph)

    base_cat = ["civil_status", "residence", "family_income", "shs_strand", "scholarship", "ethnicity", "course", "high_school"]
    for c in base_cat:
        if c not in clean_df.columns:
            clean_df[c] = "Unknown"
        clean_df[c] = clean_df[c].apply(normalize_text)

    for c in ["shs_strand", "ethnicity", "residence", "family_income", "mother_occ_group", "father_occ_group", "course"]:
        clean_df[c] = rare_group(clean_df[c], min_count=4)

    for c in ["age_first_year", "hs_average", "income_rank", "intro_computing", "programming", "gwa_1st_year"]:
        clean_df[f"{c}_missing"] = clean_df[c].isna().astype(int)

    return clean_df


ENROLLMENT_FEATURES = [
    "age_first_year", "hs_average", "income_rank", "extracurricular_count",
    "civil_status", "residence", "family_income", "shs_strand", "has_scholarship",
    "has_extracurricular", "ethnicity", "course", "mother_occ_group", "father_occ_group",
    "age_first_year_missing", "hs_average_missing", "income_rank_missing"
]

ACADEMIC_FEATURES = ENROLLMENT_FEATURES + [
    "intro_computing", "programming", "gwa_1st_year",
    "intro_performance", "programming_performance", "performance_1st_year",
    "intro_computing_missing", "programming_missing", "gwa_1st_year_missing"
]


def get_modeling_data(clean_df, feature_set, min_class_count):
    candidate = clean_df[clean_df["degree_status"].isin(["Completed", "Stopped", "Shifted"])].copy()
    counts = candidate["degree_status"].value_counts()
    usable_classes = counts[counts >= min_class_count].index.tolist()
    candidate = candidate[candidate["degree_status"].isin(usable_classes)].copy()
    features = ENROLLMENT_FEATURES if feature_set == "Enrollment-only" else ACADEMIC_FEATURES
    features = [f for f in features if f in candidate.columns]
    return candidate[features], candidate["degree_status"], features, usable_classes


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])


def safe_cv(y):
    min_count = pd.Series(y).value_counts().min()
    if min_count >= 5:
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if min_count >= 3:
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)


def make_model_zoo(n_classes):
    models = {
        "Logistic Regression Balanced": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest Balanced": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees Balanced": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Balanced Random Forest": BalancedRandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, replacement=True, sampling_strategy="all", n_jobs=-1),
        "Easy Ensemble": EasyEnsembleClassifier(n_estimators=15, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["XGBoost Conservative"] = XGBClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, objective=objective, eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=RANDOM_STATE
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM Conservative"] = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        loss = "Logloss" if n_classes == 2 else "MultiClass"
        models["CatBoost Balanced"] = CatBoostClassifier(
            iterations=150, depth=3, learning_rate=0.05, loss_function=loss,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False
        )
    except Exception:
        pass
    return models


@st.cache_data(show_spinner=False)
def evaluate_models_cached(X, y, feature_set_name):
    preprocessor = make_preprocessor(X)
    cv = safe_cv(y)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    rows = []
    n_classes = y.nunique()
    models = make_model_zoo(n_classes)
    samplers = [("No Sampling", None), ("RandomOverSampler", RandomOverSampler(random_state=RANDOM_STATE))]
    if pd.Series(y).value_counts().min() >= 3:
        k = max(1, min(3, pd.Series(y).value_counts().min() - 1))
        samplers.append(("SMOTE", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)))

    for sampler_name, sampler in samplers:
        for model_name, model in models.items():
            try:
                steps = [("preprocessor", preprocessor)]
                if sampler is not None:
                    steps.append(("sampler", sampler))
                steps.append(("model", model))
                pipe = ImbPipeline(steps)
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
                rows.append({
                    "Feature_Set": feature_set_name,
                    "Sampler": sampler_name,
                    "Model": model_name,
                    "Accuracy": float(np.mean(scores["test_accuracy"])),
                    "Balanced_Accuracy": float(np.mean(scores["test_balanced_accuracy"])),
                    "Precision_macro": float(np.mean(scores["test_precision_macro"])),
                    "Recall_macro": float(np.mean(scores["test_recall_macro"])),
                    "F1_macro": float(np.mean(scores["test_f1_macro"])),
                    "F1_macro_std": float(np.std(scores["test_f1_macro"])),
                })
            except Exception as e:
                rows.append({
                    "Feature_Set": feature_set_name, "Sampler": sampler_name, "Model": model_name,
                    "Accuracy": np.nan, "Balanced_Accuracy": np.nan, "Precision_macro": np.nan,
                    "Recall_macro": np.nan, "F1_macro": np.nan, "F1_macro_std": np.nan,
                    "Error": str(e)[:160],
                })
    return pd.DataFrame(rows).sort_values(["F1_macro", "Balanced_Accuracy"], ascending=False)


def build_pipeline(best_row, X, y):
    preprocessor = make_preprocessor(X)
    model = make_model_zoo(y.nunique())[best_row["Model"]]
    sampler_name = best_row["Sampler"]
    steps = [("preprocessor", preprocessor)]
    if sampler_name == "RandomOverSampler":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif sampler_name == "SMOTE":
        k = max(1, min(3, pd.Series(y).value_counts().min() - 1))
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def band_numeric_for_rules(df):
    out = df.copy()
    out["age_band"] = pd.cut(out["age_first_year"], bins=[0, 18, 21, 99], labels=["Age <=18", "Age 19-21", "Age >=22"])
    out["hs_average_band"] = pd.cut(out["hs_average"], bins=[0, 79, 89, 100], labels=["HS low", "HS average", "HS high"])
    out["income_band"] = out["income_rank"].map({1: "Income very low", 2: "Income low", 3: "Income moderate", 4: "Income upper moderate", 5: "Income high", 6: "Income very high"})
    return out


def run_status_association_rules(df, classes, min_support=0.04, min_confidence=0.50):
    rules_df = band_numeric_for_rules(df[df["degree_status"].isin(classes)].copy())
    cols = [
        "degree_status", "course", "shs_strand", "has_scholarship", "has_extracurricular",
        "family_income", "residence", "mother_occ_group", "father_occ_group", "performance_1st_year",
        "programming_performance", "intro_performance", "age_band", "hs_average_band", "income_band"
    ]
    cols = [c for c in cols if c in rules_df.columns]
    transactions = []
    for _, row in rules_df[cols].iterrows():
        items = []
        for col in cols:
            val = row[col]
            if pd.notna(val) and str(val) != "Unknown":
                items.append(f"{col}={val}")
        transactions.append(items)
    if not transactions:
        return pd.DataFrame()
    te = TransactionEncoder()
    basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    if len(itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    status_items = {f"degree_status={c}" for c in classes}
    rules = rules[rules["consequents"].apply(lambda x: len(set(x) & status_items) > 0)]
    if len(rules) == 0:
        return pd.DataFrame()
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    rules["antecedents_text"] = rules["antecedents"].apply(lambda x: " AND ".join(sorted(list(x))))
    rules["consequents_text"] = rules["consequents"].apply(lambda x: " AND ".join(sorted(list(x))))
    return rules[["antecedents_text", "consequents_text", "support", "confidence", "lift"]]


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Student Degree Outcome Prediction and Pattern Mining")
st.caption("A deployment-ready educational data mining dashboard for predicting and analyzing student completion, stopping, and shifting behavior.")

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload student profile Excel file", type=["xlsx", "xls"])
    st.caption("For public GitHub repositories, upload the Excel file at runtime instead of storing student data in GitHub.")
    min_class_count = st.slider("Minimum records per class for modeling", 2, 10, 2)
    feature_set = st.radio("Prediction feature set", ["Enrollment-only", "Academic-inclusive"], index=0)
    min_support = st.slider("Association rule support", 0.01, 0.20, 0.04, 0.01)
    min_confidence = st.slider("Association rule confidence", 0.30, 0.90, 0.50, 0.05)

if uploaded_file is not None:
    raw_df = load_excel(uploaded_file)
elif DEFAULT_DATA.exists():
    raw_df = load_excel(DEFAULT_DATA)
else:
    st.info("Upload an Excel file to begin. Optional: place a private default dataset at `data/StudProfile.xlsx`.")
    st.stop()

clean_df = prepare_data(raw_df)
model3_df = clean_df[clean_df["degree_status"].isin(["Completed", "Stopped", "Shifted"])].copy()

status_counts = clean_df["degree_status"].value_counts().reset_index()
status_counts.columns = ["Degree Status", "Count"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total records", f"{len(clean_df):,}")
m2.metric("Modeling records", f"{len(model3_df):,}")
m3.metric("Variables after cleaning", f"{clean_df.shape[1]:,}")
m4.metric("Usable status classes", f"{model3_df['degree_status'].nunique():,}")

tabs = st.tabs([
    "📊 Dataset Overview", "🔎 Feature Relationships", "🤖 Prediction Models",
    "🧠 Feature Influence", "🔗 Association Rules", "🌀 Class Overlap", "📑 Interpretation"
])

with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(status_counts, x="Degree Status", y="Count", title="Degree behavior status distribution", text="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Explanation:** This chart shows whether the dataset is imbalanced. If one status dominates, accuracy can become misleading, so F1-macro and balanced accuracy are prioritized.")
    with c2:
        missing = (clean_df.isna().mean().sort_values(ascending=False).head(20) * 100).reset_index()
        missing.columns = ["Feature", "Missing Percent"]
        fig = px.bar(missing, x="Missing Percent", y="Feature", orientation="h", title="Top variables with missing values")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Explanation:** High missingness may weaken model performance. The app uses median/mode imputation and missing-value indicators.")
    st.dataframe(clean_df.head(30), use_container_width=True)

with tabs[1]:
    st.subheader("Feature relationships with degree outcome")
    feature_options = [
        "course", "shs_strand", "has_scholarship", "has_extracurricular", "family_income", "residence",
        "mother_occ_group", "father_occ_group", "performance_1st_year", "programming_performance", "intro_performance"
    ]
    selected_feature = st.selectbox("Select feature", [f for f in feature_options if f in model3_df.columns])
    ct = pd.crosstab(model3_df[selected_feature], model3_df["degree_status"], normalize="index") * 100
    ct_plot = ct.reset_index().melt(id_vars=selected_feature, var_name="Degree Status", value_name="Percent")
    fig = px.bar(ct_plot, x=selected_feature, y="Percent", color="Degree Status", barmode="stack", title=f"Degree status by {selected_feature}")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Explanation:** Taller sections show which student outcomes are more common within each feature category. This supports feature relationship analysis for Chapter 4.")
    st.dataframe(ct.round(2), use_container_width=True)

    numeric_options = ["age_first_year", "hs_average", "income_rank", "intro_computing", "programming", "gwa_1st_year", "extracurricular_count"]
    numeric_options = [f for f in numeric_options if f in model3_df.columns]
    num_feature = st.selectbox("Select numeric feature", numeric_options)
    fig = px.box(model3_df, x="degree_status", y=num_feature, points="all", title=f"{num_feature} by degree behavior status")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Explanation:** Overlapping boxes suggest weak separation among classes. This helps explain why model accuracy may remain limited.")

with tabs[2]:
    st.subheader("Imbalance-aware prediction models")
    X, y, features, usable_classes = get_modeling_data(clean_df, feature_set, min_class_count)
    st.write("Class distribution used for modeling:", dict(Counter(y)))
    if y.nunique() < 2:
        st.error("At least two classes are required for modeling. Lower the minimum class count or check the target field.")
    else:
        with st.spinner("Training and cross-validating models..."):
            results = evaluate_models_cached(X, y, feature_set)
        st.dataframe(results, use_container_width=True)
        plot_df = results.dropna(subset=["F1_macro"]).head(12).copy()
        if len(plot_df):
            plot_df["Model Label"] = plot_df["Sampler"] + " | " + plot_df["Model"]
            fig = px.bar(plot_df.iloc[::-1], x="F1_macro", y="Model Label", orientation="h", title="Top models by F1-macro")
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Explanation:** F1-macro is emphasized because it treats minority classes more fairly than accuracy in imbalanced datasets.")

        best = results.dropna(subset=["F1_macro"]).iloc[0]
        pipe = build_pipeline(best, X, y)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics_df = pd.DataFrame([{
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision Macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall Macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 Macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }])
        st.write("Best model:", f"**{best['Sampler']} | {best['Model']}**")
        st.dataframe(metrics_df, use_container_width=True)
        cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
        fig = go.Figure(data=go.Heatmap(z=cm, x=pipe.classes_, y=pipe.classes_, colorscale="Blues", text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Best model confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["model_objects"] = (pipe, X, y, X_test, y_test, y_pred, best, results)
        st.write("**Explanation:** The confusion matrix shows which outcomes are often misclassified. Frequent confusion means the classes are overlapping or predictors are weak.")

with tabs[3]:
    st.subheader("Feature influence analysis")
    if "model_objects" not in st.session_state:
        st.info("Run the Prediction Models tab first.")
    else:
        pipe, X, y, X_test, y_test, y_pred, best, results = st.session_state["model_objects"]
        with st.spinner("Computing permutation importance..."):
            try:
                perm = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1)
                importance_df = pd.DataFrame({
                    "Feature": X_test.columns,
                    "Importance": perm.importances_mean,
                    "Std": perm.importances_std,
                }).sort_values("Importance", ascending=False)
                st.dataframe(importance_df, use_container_width=True)
                top = importance_df.head(15).iloc[::-1]
                fig = px.bar(top, x="Importance", y="Feature", orientation="h", title="Top feature influence using permutation importance")
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Explanation:** Higher importance means the model performance drops more when that feature is shuffled. These are the strongest available indicators of completion/stopping/shifting.")
            except Exception as e:
                st.warning(f"Permutation importance could not be computed: {e}")

with tabs[4]:
    st.subheader("Association rules for degree behavior")
    X, y, features, usable_classes = get_modeling_data(clean_df, feature_set, min_class_count)
    rules = run_status_association_rules(clean_df, usable_classes, min_support=min_support, min_confidence=min_confidence)
    if len(rules) == 0:
        st.warning("No rules found. Try lowering support or confidence.")
    else:
        st.dataframe(rules.head(50), use_container_width=True)
        top_rules = rules.head(12).copy()
        top_rules["Rule"] = top_rules["antecedents_text"].str.slice(0, 70) + " → " + top_rules["consequents_text"].str.replace("degree_status=", "")
        fig = px.bar(top_rules.iloc[::-1], x="lift", y="Rule", orientation="h", title="Top status association rules by lift")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Explanation:** Lift above 1 means the pattern is more strongly associated with the outcome than random chance.")

with tabs[5]:
    st.subheader("Class overlap visualization")
    X, y, features, usable_classes = get_modeling_data(clean_df, feature_set, min_class_count)
    if y.nunique() < 2:
        st.warning("Need at least two classes for overlap visualization.")
    else:
        pre = make_preprocessor(X)
        X_processed = pre.fit_transform(X)
        X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else np.asarray(X_processed)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X_dense)
        pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "degree_status": y.values})
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="degree_status", title="PCA visualization of class overlap")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Explanation:** If points from different classes overlap heavily, the dataset has weak class separability. In that case, lower accuracy is a data limitation, not only a model problem.")

with tabs[6]:
    st.subheader("Research interpretation guide")
    st.markdown(
        """
        **Dataset condition.** The model should be interpreted with caution when the class distribution is imbalanced, the sample size is small, and the PCA plot shows overlapping classes.

        **Objective 1: Identify influential features.** Use the Feature Relationships and Feature Influence tabs. Report the highest permutation-importance variables as exploratory indicators, not causal factors.

        **Objective 2: Build prediction models.** Use the Prediction Models tab. Report F1-macro and balanced accuracy together with accuracy because ordinary accuracy can be misleading under imbalance.

        **Objective 3: Analyze patterns and relationships.** Use the Association Rules and Class Overlap tabs. Association rules explain common feature combinations linked to completion, stopping, or shifting.

        **Recommended statement.** The system is best positioned as an exploratory educational data mining and early-warning dashboard. It supports student profiling and decision support, but stronger validation requires a larger and more balanced dataset.
        """
    )

csv = clean_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download cleaned dataset", csv, "cleaned_student_dataset.csv", "text/csv")

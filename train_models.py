from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from preprocessing import RANDOM_STATE, load_excel, clean_student_data, get_modeling_data, TARGET

DATA_PATH = Path("data/StudProfile.xlsx")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])


def safe_cv(y):
    min_count = pd.Series(y).value_counts().min()
    n_splits = int(min(5, max(2, min_count)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def model_zoo(y):
    return {
        "Logistic Regression Balanced": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest Balanced": RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees Balanced": ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "Balanced Random Forest": BalancedRandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, replacement=True, sampling_strategy="all", n_jobs=1),
        "Easy Ensemble": EasyEnsembleClassifier(n_estimators=8, random_state=RANDOM_STATE, n_jobs=1),
    }


def build_pipeline(model_name, sampler_name, X, y):
    preprocessor = make_preprocessor(X)
    model = model_zoo(y)[model_name]
    steps = [("preprocessor", preprocessor)]
    if sampler_name == "RandomOverSampler":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif sampler_name == "SMOTE":
        k = max(1, min(3, pd.Series(y).value_counts().min() - 1))
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def train_all(data_path=DATA_PATH, feature_set="academic"):
    raw_df = load_excel(data_path)
    clean_df = clean_student_data(raw_df)
    X, y, features, modeling_df, classes = get_modeling_data(clean_df, feature_set=feature_set)

    samplers = ["No Sampling", "RandomOverSampler"]
    if pd.Series(y).value_counts().min() >= 3:
        samplers.append("SMOTE")

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    rows = []
    cv = safe_cv(y)
    for sampler_name in samplers:
        for model_name in model_zoo(y):
            try:
                pipe = build_pipeline(model_name, sampler_name, X, y)
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
                rows.append({
                    "Sampler": sampler_name,
                    "Model": model_name,
                    "Accuracy_mean": float(np.mean(scores["test_accuracy"])),
                    "Balanced_Accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
                    "Precision_macro_mean": float(np.mean(scores["test_precision_macro"])),
                    "Recall_macro_mean": float(np.mean(scores["test_recall_macro"])),
                    "F1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                    "F1_macro_std": float(np.std(scores["test_f1_macro"])),
                })
            except Exception as e:
                rows.append({"Sampler": sampler_name, "Model": model_name, "Error": str(e)})

    results = pd.DataFrame(rows).sort_values(["F1_macro_mean", "Balanced_Accuracy_mean"], ascending=False, na_position="last")
    best = results.dropna(subset=["F1_macro_mean"]).iloc[0]
    best_pipe = build_pipeline(best["Model"], best["Sampler"], X, y)
    best_pipe.fit(X, y)

    joblib.dump(best_pipe, MODELS_DIR / "outcome_model.pkl")
    joblib.dump(features, MODELS_DIR / "feature_list.pkl")
    metadata = {
        "feature_set": feature_set,
        "target": TARGET,
        "classes": list(classes),
        "best_model": str(best["Model"]),
        "best_sampler": str(best["Sampler"]),
        "n_records_used": int(len(y)),
        "class_distribution": pd.Series(y).value_counts().to_dict(),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    clean_df.to_csv(OUTPUTS_DIR / "cleaned_student_dataset.csv", index=False)
    results.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    return results, metadata


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError("Place your Excel file at data/StudProfile.xlsx, then run: python train_models.py")
    results, metadata = train_all(DATA_PATH, feature_set="academic")
    print("Training complete.")
    print(metadata)
    print(results.head(10))

"""Train and evaluate customer churn models for the Bengkel Koding UAS project.

This script follows the scoring requirements:
1. EDA support outputs.
2. Three model categories: conventional, ensemble bagging, and ensemble voting.
3. Three experiment scenarios: direct, preprocessing, and hyperparameter tuning.
4. Best model export for Streamlit deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from model_utils import (
    FEATURE_EXPLANATIONS,
    IQRClipper,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    align_features,
    engineer_customer_features,
    normalize_column_names,
    split_feature_types,
)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder that works on old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate classification metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    return metrics


def metrics_to_row(scenario: str, model_name: str, metrics: dict) -> dict:
    """Convert a metric dictionary into one row for the summary table."""
    return {
        "scenario": scenario,
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }


def build_direct_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Minimal adapter for direct modelling without cleaning, scaling, or tuning."""
    numeric_features, categorical_features = split_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_preprocessor(X: pd.DataFrame, use_outlier_clipper: bool = True) -> ColumnTransformer:
    """Create preprocessing pipeline for clean modelling and deployment."""
    numeric_features, categorical_features = split_feature_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_outlier_clipper:
        numeric_steps.append(("outlier_clipper", IQRClipper(factor=1.5)))
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer, class_weight: str | None = None) -> Dict[str, Pipeline]:
    """Create three required model categories."""
    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=800,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=40,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    voting = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                VotingClassifier(
                    estimators=[
                        (
                            "lr",
                            LogisticRegression(
                                max_iter=800,
                                solver="liblinear",
                                random_state=RANDOM_STATE,
                                class_weight=class_weight,
                            ),
                        ),
                        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=class_weight)),
                        ("nb", GaussianNB()),
                    ],
                    voting="soft",
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "Voting Classifier": voting,
    }


def run_experiment(
    scenario: str,
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[dict], dict]:
    """Fit every model and return metric rows with fitted estimators."""
    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario} | Model: {model_name}")
        print(f"{'=' * 80}")

        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)
        rows.append(metrics_to_row(scenario, model_name, metrics))
        fitted_models[model_name] = {"estimator": model, "metrics": metrics}

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print("Confusion matrix:")
        print(np.asarray(metrics["confusion_matrix"]))

    return rows, fitted_models


def aggregate_feature_importance(model: Pipeline, categorical_features: Iterable[str]) -> pd.DataFrame:
    """Aggregate transformed feature importance back to original feature names."""
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "importance": rf_model.feature_importances_,
        }
    )

    categorical_features = list(categorical_features)

    def get_original_feature(feature_name: str) -> str:
        if feature_name.startswith("num__"):
            return feature_name.replace("num__", "", 1)
        if feature_name.startswith("cat__"):
            clean_name = feature_name.replace("cat__", "", 1)
            for column in categorical_features:
                if clean_name == column or clean_name.startswith(f"{column}_"):
                    return column
            return clean_name
        return feature_name

    importance_df["feature"] = importance_df["transformed_feature"].apply(get_original_feature)
    original_importance = (
        importance_df.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    original_importance["cumulative_importance"] = original_importance["importance"].cumsum()
    return original_importance


def select_important_features(feature_importance: pd.DataFrame) -> list[str]:
    """Select features using mean importance and a minimum number of predictors."""
    threshold = feature_importance["importance"].mean()
    selected_features = feature_importance.loc[
        feature_importance["importance"] >= threshold, "feature"
    ].tolist()

    minimum_features = min(10, len(feature_importance))
    if len(selected_features) < minimum_features:
        selected_features = feature_importance.head(minimum_features)["feature"].tolist()

    return selected_features


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[dict, pd.DataFrame]:
    """Tune every required model category using RandomizedSearchCV."""
    models = build_models(preprocessor, class_weight="balanced")
    parameter_spaces = {
        "Logistic Regression": {
            "model__C": [0.1, 0.5, 1, 3],
            "model__solver": ["liblinear"],
        },
        "Random Forest": {
            "model__n_estimators": [50, 80],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", None],
        },
        "Voting Classifier": {
            "model__lr__C": [0.5, 1],
            "model__dt__max_depth": [None, 6, 10],
            "model__dt__min_samples_leaf": [1, 3, 5],
            "model__weights": [(1, 1, 1), (2, 1, 1)],
        },
    }

    best_estimators = {}
    tuning_rows = []

    for model_name, model in models.items():
        param_grid = list(ParameterGrid(parameter_spaces[model_name]))
        n_iter = min(3, len(param_grid))

        print(f"\n{'=' * 80}")
        print(f"Hyperparameter tuning: {model_name}")
        print(f"{'=' * 80}")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_spaces[model_name],
            n_iter=n_iter,
            scoring="f1",
            cv=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_estimators[model_name] = search.best_estimator_
        tuning_rows.append(
            {
                "model": model_name,
                "best_cv_f1": search.best_score_,
                "best_params": search.best_params_,
            }
        )

        print(f"Best CV F1-score: {search.best_score_:.4f}")
        print("Best parameters:")
        print(search.best_params_)

    return best_estimators, pd.DataFrame(tuning_rows)


def get_numeric_outlier_summary(X_train: pd.DataFrame) -> pd.DataFrame:
    """Summarise outliers on training data using IQR."""
    numeric_columns = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    rows = []

    for column in numeric_columns:
        values = X_train[column].dropna()
        unique_values = set(values.unique())
        if unique_values.issubset({0, 1}):
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((X_train[column] < lower) | (X_train[column] > upper)).sum()
        rows.append(
            {
                "feature": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count_train": int(outlier_count),
                "outlier_percentage_train": round(outlier_count / len(X_train) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_percentage_train", ascending=False)


def build_metadata(
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    selected_features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    best_model_name: str,
    best_metrics: dict,
    best_params: dict,
    results_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> dict:
    """Create deployment metadata used by Streamlit."""
    feature_ranges = {}
    for column in df_model.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns:
        if column == TARGET_COL:
            continue
        feature_ranges[column] = {
            "min": float(df_model[column].min(skipna=True)),
            "median": float(df_model[column].median(skipna=True)),
            "max": float(df_model[column].max(skipna=True)),
        }

    categorical_options = {}
    for column in df_raw.select_dtypes(include=["object"]).columns:
        options = sorted(df_raw[column].dropna().astype(str).unique().tolist())
        categorical_options[column] = options

    for column in df_model.select_dtypes(include=["object"]).columns:
        options = sorted(df_model[column].dropna().astype(str).unique().tolist())
        categorical_options[column] = options

    return {
        "target_col": TARGET_COL,
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model": best_model_name,
        "best_params": best_params,
        "best_scenario": str(results_df.sort_values("f1_score", ascending=False).iloc[0]["scenario"]),
        "metrics": best_metrics,
        "evaluation_results": results_df.to_dict(orient="records"),
        "feature_importance": feature_importance_df.head(20).to_dict(orient="records"),
        "feature_ranges": feature_ranges,
        "categorical_options": categorical_options,
        "feature_explanations": FEATURE_EXPLANATIONS,
        "reference_date": str(reference_date.date()),
        "class_distribution": df_raw[TARGET_COL].value_counts().sort_index().to_dict(),
        "missing_percentage": (df_raw.isnull().mean() * 100).round(2).to_dict(),
        "required_raw_columns": [column for column in df_raw.columns if column != TARGET_COL],
    }


def main():
    parser = argparse.ArgumentParser(description="Train customer churn models and export the best estimator.")
    parser.add_argument("--data", default="Sales - Marketing customer dataset.csv", help="Path dataset CSV.")
    parser.add_argument("--model-output", default="best_model.joblib", help="Output model path.")
    parser.add_argument("--metadata-output", default="model_metadata.joblib", help="Output metadata path.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for evaluation tables.")
    args = parser.parse_args()

    data_path = Path(args.data)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df_raw = normalize_column_names(pd.read_csv(data_path))
    if TARGET_COL not in df_raw.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan pada dataset.")

    print("Dataset shape:", df_raw.shape)
    print("Target distribution:")
    print(df_raw[TARGET_COL].value_counts())

    df_raw.drop_duplicates().to_csv(artifacts_dir / "dataset_after_duplicate_check.csv", index=False)
    (df_raw.isnull().mean() * 100).sort_values(ascending=False).to_csv(
        artifacts_dir / "missing_value_percentage.csv", header=["missing_percentage"]
    )

    # Scenario 1: direct modelling.
    X_direct = df_raw.drop(columns=[TARGET_COL])
    y = df_raw[TARGET_COL]
    X_train_direct, X_test_direct, y_train, y_test = train_test_split(
        X_direct,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    direct_models = build_models(build_direct_preprocessor(X_train_direct), class_weight="balanced")
    direct_rows, direct_fitted = run_experiment(
        "Direct Modeling", direct_models, X_train_direct, X_test_direct, y_train, y_test
    )

    # Scenario 2: cleaned preprocessing.
    df_model = engineer_customer_features(df_raw)
    df_model = df_model.drop_duplicates().reset_index(drop=True)
    X_clean = df_model.drop(columns=[TARGET_COL])
    y_clean = df_model[TARGET_COL]
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean,
        y_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clean,
    )

    outlier_summary = get_numeric_outlier_summary(X_train_clean)
    outlier_summary.to_csv(artifacts_dir / "outlier_summary_train.csv", index=False)

    preprocessing_models = build_models(build_preprocessor(X_train_clean), class_weight="balanced")
    preprocessing_rows, preprocessing_fitted = run_experiment(
        "Preprocessing", preprocessing_models, X_train_clean, X_test_clean, y_train_clean, y_test_clean
    )

    # Feature importance before tuning.
    rf_importance_model = preprocessing_fitted["Random Forest"]["estimator"]
    _, categorical_features_clean = split_feature_types(X_train_clean)
    feature_importance = aggregate_feature_importance(rf_importance_model, categorical_features_clean)
    selected_features = select_important_features(feature_importance)
    feature_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    print("\nSelected features:")
    print(selected_features)

    X_train_selected = align_features(X_train_clean, selected_features)
    X_test_selected = align_features(X_test_clean, selected_features)
    numeric_selected, categorical_selected = split_feature_types(X_train_selected)

    # Scenario 3: feature selection + hyperparameter tuning.
    selected_preprocessor = build_preprocessor(X_train_selected)
    best_estimators, tuning_results = tune_models(X_train_selected, y_train_clean, selected_preprocessor)
    tuning_results.to_csv(artifacts_dir / "tuning_results.csv", index=False)

    tuning_rows = []
    tuning_metrics_by_model = {}
    for model_name, estimator in best_estimators.items():
        metrics = evaluate_classifier(estimator, X_test_selected, y_test_clean)
        tuning_rows.append(metrics_to_row("Hyperparameter Tuning + Feature Selection", model_name, metrics))
        tuning_metrics_by_model[model_name] = metrics
        print(f"\nTuned evaluation: {model_name}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(np.asarray(metrics["confusion_matrix"]))

    results_df = pd.DataFrame(direct_rows + preprocessing_rows + tuning_rows)
    results_df.to_csv(artifacts_dir / "model_evaluation_results.csv", index=False)

    best_row = results_df.sort_values("f1_score", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_scenario = best_row["scenario"]

    if best_scenario == "Hyperparameter Tuning + Feature Selection":
        best_model = best_estimators[best_model_name]
        best_metrics = tuning_metrics_by_model[best_model_name]
        best_params = tuning_results.loc[tuning_results["model"] == best_model_name, "best_params"].iloc[0]
        final_features = selected_features
        final_numeric = numeric_selected
        final_categorical = categorical_selected
    elif best_scenario == "Preprocessing":
        best_model = preprocessing_fitted[best_model_name]["estimator"]
        best_metrics = preprocessing_fitted[best_model_name]["metrics"]
        best_params = best_model.named_steps["model"].get_params()
        final_features = X_train_clean.columns.tolist()
        final_numeric, final_categorical = split_feature_types(X_train_clean)
    else:
        best_model = direct_fitted[best_model_name]["estimator"]
        best_metrics = direct_fitted[best_model_name]["metrics"]
        best_params = best_model.named_steps["model"].get_params()
        final_features = X_train_direct.columns.tolist()
        final_numeric, final_categorical = split_feature_types(X_train_direct)

    reference_date = pd.to_datetime(df_raw["last_purchase_date"], errors="coerce").max()
    metadata = build_metadata(
        df_raw=df_raw,
        df_model=df_model,
        selected_features=final_features,
        numeric_features=final_numeric,
        categorical_features=final_categorical,
        best_model_name=best_model_name,
        best_metrics=best_metrics,
        best_params=best_params,
        results_df=results_df,
        feature_importance_df=feature_importance,
        reference_date=reference_date,
    )

    # Save JSON version for inspection and joblib version for Streamlit.
    with open(artifacts_dir / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)

    joblib.dump(best_model, args.model_output)
    joblib.dump(metadata, args.metadata_output)

    print("\nBest model exported")
    print("Scenario:", best_scenario)
    print("Model   :", best_model_name)
    print("F1-score:", f"{best_metrics['f1_score']:.4f}")
    print("Model path:", args.model_output)
    print("Metadata path:", args.metadata_output)


if __name__ == "__main__":
    main()

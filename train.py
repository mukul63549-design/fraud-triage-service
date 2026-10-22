
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from prepare import clean_transactions, add_features


def load_and_split():
    raw = pd.read_csv("transactions_raw.csv")
    df = add_features(clean_transactions(raw))

    excluded = [
        "txn_id",
        "customer_id",
        "chargeback_filed",
        "is_fraud",
    ]

    X = pd.get_dummies(
        df.drop(columns=excluded),
        dtype=int,
    )
    y = df["is_fraud"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    median = X_train["distance_from_home_km"].median()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for split in [X_train, X_val, X_test]:
        split["distance_from_home_km"] = (
            split["distance_from_home_km"].fillna(median)
        )

    print("\n=== SPLIT CHECKPOINTS ===")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))
    print("Training fraud rows:", int(y_train.sum()))
    print("Validation fraud rows:", int(y_val.sum()))
    print("Test fraud rows:", int(y_test.sum()))
    print("Training distance median:", median)
    print("Encoded input columns:", len(X.columns))
    print("Input column order:")
    print(list(X.columns))

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        median, list(X.columns),
    )


def make_pipeline(class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
            ),
        ),
    ])


def print_metrics(name, y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(
        f"{name}: "
        f"precision={precision_score(y_true, predictions, zero_division=0):.3f}, "
        f"recall={recall_score(y_true, predictions, zero_division=0):.3f}, "
        f"F1={f1_score(y_true, predictions, zero_division=0):.3f}, "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


def compare_imbalance_models(X_train, X_val, y_train, y_val):
    print("\n=== IMBALANCE MODEL COMPARISON (THRESHOLD 0.5) ===")

    plain = make_pipeline()
    plain.fit(X_train, y_train)
    plain_pred = (
        plain.predict_proba(X_val)[:, 1] >= 0.5
    ).astype(int)
    print_metrics("Plain", y_val, plain_pred)

    balanced = make_pipeline(class_weight="balanced")
    balanced.fit(X_train, y_train)
    balanced_pred = (
        balanced.predict_proba(X_val)[:, 1] >= 0.5
    ).astype(int)
    print_metrics("Balanced", y_val, balanced_pred)

    training = X_train.copy()
    training["is_fraud"] = y_train.to_numpy()

    majority = training[training["is_fraud"] == 0]
    minority = training[training["is_fraud"] == 1]

    minority_up = minority.sample(
        n=len(majority),
        replace=True,
        random_state=42,
    )

    oversampled = pd.concat(
        [majority, minority_up],
        ignore_index=True,
    ).sample(frac=1, random_state=42)

    y_over = oversampled.pop("is_fraud")
    X_over = oversampled

    print("Oversampled training rows:", len(X_over))
    print("Oversampled non-fraud rows:", int((y_over == 0).sum()))
    print("Oversampled fraud rows:", int((y_over == 1).sum()))

    over_model = make_pipeline()
    over_model.fit(X_over, y_over)

    over_pred = (
        over_model.predict_proba(X_val)[:, 1] >= 0.5
    ).astype(int)

    print_metrics("Oversampled", y_val, over_pred)


def cross_validate_and_tune(X_train, y_train):
    print("\n=== CROSS-VALIDATION AND RANDOM FOREST TUNING ===")

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    logistic = make_pipeline(class_weight="balanced")
    logistic_scores = cross_val_score(
        logistic,
        X_train,
        y_train,
        cv=folds,
        scoring="f1",
    )

    print(
        "Balanced Logistic Regression CV F1: "
        f"mean={logistic_scores.mean():.3f}, "
        f"std={logistic_scores.std():.3f}"
    )

    forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )

    forest_scores = cross_val_score(
        forest,
        X_train,
        y_train,
        cv=folds,
        scoring="f1",
    )

    print(
        "Untuned Random Forest CV F1: "
        f"mean={forest_scores.mean():.3f}, "
        f"std={forest_scores.std():.3f}"
    )

    grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    }

    search = GridSearchCV(
        estimator=RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid=grid,
        scoring="f1",
        cv=folds,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print("Grid combinations:", len(search.cv_results_["params"]))
    print("Grid folds:", folds.get_n_splits())
    print("Best parameters:", search.best_params_)
    print(f"Best mean CV F1: {search.best_score_:.3f}")

    return search.best_estimator_, search.best_params_


def select_threshold(final_model, X_val, y_val):
    print("\n=== VALIDATION THRESHOLD SELECTION ===")

    probabilities = final_model.predict_proba(X_val)[:, 1]

    best_threshold = None
    best_recall = -1

    for threshold in [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9,
    ]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        print(
            f"Threshold={threshold:.1f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}"
        )

        if recall >= 0.65 and threshold > (
            best_threshold if best_threshold is not None else -1
        ):
            best_threshold = threshold
            best_recall = recall

    if best_threshold is None:
        raise RuntimeError(
            "No threshold achieved the required recall of 0.65"
        )

    print(f"Selected threshold: {best_threshold:.1f}")
    print(f"Selected threshold recall: {best_recall:.3f}")

    return best_threshold


def evaluate_final_test(final_model, X_test, y_test, threshold):
    print("\n=== FINAL TEST EVALUATION ===")

    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("Test threshold:", threshold)
    print_metrics("Final Test", y_test, predictions)


def run_leakage_experiment(
    X_train, X_val, y_train, y_val, best_params
):
    print("\n=== DATA LEAKAGE EXPERIMENT ===")

    raw = pd.read_csv("transactions_raw.csv")
    df = add_features(clean_transactions(raw))

    # Deliberately include chargeback_filed to demonstrate leakage.
    # This field is not available for safe real-time fraud prediction.
    excluded = ["txn_id", "customer_id", "is_fraud"]

    X_leak = pd.get_dummies(
        df.drop(columns=excluded),
        dtype=int,
    )

    X_leak_train = X_leak.loc[X_train.index].copy()
    X_leak_val = X_leak.loc[X_val.index].copy()

    median = X_leak_train["distance_from_home_km"].median()

    X_leak_train["distance_from_home_km"] = (
        X_leak_train["distance_from_home_km"].fillna(median)
    )
    X_leak_val["distance_from_home_km"] = (
        X_leak_val["distance_from_home_km"].fillna(median)
    )

    leak_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        **best_params,
    )

    leak_model.fit(X_leak_train, y_train)

    predictions = (
        leak_model.predict_proba(X_leak_val)[:, 1] >= 0.5
    ).astype(int)

    print(
        "Validation F1 with leakage at threshold 0.5:",
        f"{f1_score(y_val, predictions):.3f}",
    )
    print(
        "WARNING: chargeback_filed is a post-outcome field "
        "and must not be used for real-time prediction."
    )


def print_feature_importance(final_model, columns):
    print("\n=== TOP 5 FEATURE IMPORTANCE ===")

    importances = final_model.feature_importances_

    ranked = sorted(
        zip(columns, importances),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in ranked[:5]:
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        median, columns,
    ) = load_and_split()

    compare_imbalance_models(
        X_train, X_val, y_train, y_val
    )

    final_model, best_params = cross_validate_and_tune(
        X_train, y_train
    )

    final_model.fit(X_train, y_train)

    threshold = select_threshold(
        final_model, X_val, y_val
    )

    print("\n=== PART B CHECKPOINT ===")
    print("Best parameters:", best_params)
    print("Selected threshold:", threshold)

    evaluate_final_test(
        final_model, X_test, y_test, threshold
    )

    run_leakage_experiment(
        X_train, X_val, y_train, y_val, best_params
    )

    print_feature_importance(final_model, columns)
import os
import time
import warnings
import joblib
import optuna
import mlflow
import mlflow.sklearn
from optuna.integration.mlflow import MLflowCallback

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score


warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

df = pd.read_csv(r"reviews_data\processed\cleaned_data.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["clean_text"].str.strip() != ""]
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = Pipeline(
    [
        ('tfidf' , TfidfVectorizer()),
        ('model',LogisticRegression())
    ]
)

def objective_logreg(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 5000, 30000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        tfidf__min_df=trial.suggest_int("min_df", 2, 10),
        model=LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear"
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="f1"
    ).mean()

    return score

def objective_svm(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 5000, 30000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        tfidf__min_df=trial.suggest_int("min_df", 2, 10),
        model=LinearSVC(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            class_weight="balanced"
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="f1"
    ).mean()

    return score

def objective_nb(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 5000, 30000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        tfidf__min_df=trial.suggest_int("min_df", 2, 10),
        model=MultinomialNB(
            alpha=trial.suggest_float("alpha", 0.1, 2.0)
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="f1"
    ).mean()

    return score

objectives = {
    "LogisticRegression": {
        "objective": objective_logreg,
        "experiment": "Flipkart_Sentiment_LogisticRegression"
    },
    "LinearSVM": {
        "objective": objective_svm,
        "experiment": "Flipkart_Sentiment_SVM"
    },
    "NaiveBayes": {
        "objective": objective_nb,
        "experiment": "Flipkart_Sentiment_NaiveBayes"
    }
}

mlflow.set_experiment("Flipkart_Sentiment_MultiModel")

results = {}

for model_name, cfg in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")
    mlflow.set_experiment(cfg["experiment"])

    mlflow_cb = MLflowCallback(
        metric_name="cv_f1",
        mlflow_kwargs={"nested": True}
    )

    study = optuna.create_study(direction="maximize")

    start_opt = time.time()
    study.optimize(cfg["objective"], n_trials=25, callbacks=[mlflow_cb])
    opt_time = time.time() - start_opt

    best_params = study.best_params
    best_cv = study.best_value

    # Rebuild pipeline with best params
    if model_name == "LogisticRegression":
        pipeline.set_params(
            tfidf__max_features=best_params["max_features"],
            tfidf__ngram_range=best_params["ngram_range"],
            tfidf__min_df=best_params["min_df"],
            model=LogisticRegression(
                C=best_params["C"],
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear"
            )
        )
    elif model_name == "LinearSVM":
        pipeline.set_params(
            tfidf__max_features=best_params["max_features"],
            tfidf__ngram_range=best_params["ngram_range"],
            tfidf__min_df=best_params["min_df"],
            model=LinearSVC(
                C=best_params["C"],
                class_weight="balanced"
            )
        )
    elif model_name == "NaiveBayes":
        pipeline.set_params(
            tfidf__max_features=best_params["max_features"],
            tfidf__ngram_range=best_params["ngram_range"],
            tfidf__min_df=best_params["min_df"],
            model=MultinomialNB(alpha=best_params["alpha"])
        )

    # -------------------- Train & Evaluate --------------------
    start_fit = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    start_test = time.time()
    y_test_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    y_train_pred = pipeline.predict(X_train)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # -------------------- Model Size --------------------
    os.makedirs("tmp_models", exist_ok=True)
    tmp_path = f"tmp_models/{model_name}.pkl"
    joblib.dump(pipeline, tmp_path)
    model_size = os.path.getsize(tmp_path)
    os.remove(tmp_path)

    # -------------------- MLflow Logging --------------------
    with mlflow.start_run(run_name=f"{model_name}_FINAL"):
        mlflow.log_params(best_params)

        mlflow.log_metric("cv_f1", best_cv)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        mlflow.log_metric("fit_time_sec", fit_time)
        mlflow.log_metric("test_time_sec", test_time)
        mlflow.log_metric("model_size_bytes", model_size)

        mlflow.sklearn.log_model(pipeline, "model")

    results[model_name] = {
        "cv_f1": best_cv,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "fit_time": fit_time,
        "test_time": test_time,
        "model_size": model_size
    }

# -------------------- Final Summary --------------------
print("\n================ FINAL SUMMARY ================")
for model, r in results.items():
    print(
        f"\n{model}"
        f"\n  CV F1          : {r['cv_f1']:.4f}"
        f"\n  Train F1       : {r['train_f1']:.4f}"
        f"\n  Test F1        : {r['test_f1']:.4f}"
        f"\n  Train Accuracy : {r['train_acc']:.4f}"
        f"\n  Test Accuracy  : {r['test_acc']:.4f}"
        f"\n  Fit Time (s)   : {r['fit_time']:.2f}"
        f"\n  Test Time (s)  : {r['test_time']:.4f}"
        f"\n  Model Size (B) : {r['model_size']}"
    )
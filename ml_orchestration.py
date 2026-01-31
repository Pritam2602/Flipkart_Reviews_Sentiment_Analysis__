import os
import time
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from prefect import task, flow

@task
def load_data(file_path):
    return pd.read_csv(file_path)

@task
def split_inputs_output(data, text_col, target_col):
    X = data[text_col]
    y = data[target_col]
    return X, y

@task
def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


@task
def build_model(hyperparameters):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=hyperparameters["max_features"],
            ngram_range=hyperparameters["ngram_range"],
            min_df=hyperparameters["min_df"]
        )),
        ("model", MultinomialNB(alpha=hyperparameters["alpha"]))
    ])
    return pipeline

@task
def train_model(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    return model, train_time

@task
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    return train_acc, test_acc, test_f1, test_time

@task
def log_to_mlflow(model, hyperparameters, train_acc, test_acc, test_f1,
                  train_time, test_time):

    mlflow.set_experiment("Flipkart_Sentiment_Prefect")

    with mlflow.start_run(run_name="NaiveBayes_Prefect_Run"):

        # Log parameters
        for k, v in hyperparameters.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)

        # Save model
        joblib.dump(model, "naive_bayes_model.joblib")
        model_size = os.path.getsize("naive_bayes_model.joblib") / (1024 * 1024)
        mlflow.log_metric("model_size_mb", model_size)

        # Log artifacts
        mlflow.log_artifact("naive_bayes_model.joblib")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="Flipkart_Sentiment_Best_Model"
        )

@flow(name="Flipkart Sentiment Training Flow")
def workflow():

    DATA_PATH = r"reviews_data\processed\cleaned_data.csv"
    TEXT_COL = "clean_text"
    TARGET_COL = "sentiment"

    HYPERPARAMETERS = {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "alpha": 0.5
    }

    # Load data
    data = load_data(DATA_PATH)

    # Inputs & Output
    X, y = split_inputs_output(data, TEXT_COL, TARGET_COL)

    # Train-Test Split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Build Model
    model = build_model(HYPERPARAMETERS)

    # Train Model
    model, train_time = train_model(model, X_train, y_train)

    # Evaluate
    train_acc, test_acc, test_f1, test_time = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Test F1:", test_f1)

    # Log to MLflow
    log_to_mlflow(
        model,
        HYPERPARAMETERS,
        train_acc,
        test_acc,
        test_f1,
        train_time,
        test_time
    )

if __name__ == "__main__":
    workflow.serve(
        name="flipkart-sentiment-deployment",
        cron="0 */6 * * *"   # every 6 hours
    )


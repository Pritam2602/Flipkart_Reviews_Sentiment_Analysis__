📌 Project Overview

This project focuses on sentiment analysis of Flipkart product reviews using traditional Machine Learning techniques combined with experiment tracking, hyperparameter optimization, model registry, and workflow orchestration.

The project is designed to demonstrate production-ready ML practices, including:
1. Reproducible experiments
2. Model comparison
3. Automated training pipelines
4. Model governance using MLflow
5. Orchestration using Prefect
6. Backend–Frontend integration (Flask + React)

🎯 Objectives

1. Build a robust sentiment classification model
2. Handle class imbalance in real-world review data
3. Compare multiple ML algorithms
4. Track experiments and metrics using MLflow
5. Tune hyperparameters using Optuna
6. Register and manage the best model
7. Orchestrate ML pipelines using Prefect
8. Prepare the project for deployment with Flask & React

📂 Dataset Description

The dataset consists of Flipkart product reviews from three different product categories:

Products

🏸 Badminton
🍳 Tawa
🍵 Tea

🧹 Data Preprocessing

Steps performed:
1. Lowercasing text
2. Removing punctuation & numbers
3. Removing stopwords
4. Lemmatization
5. Handling missing values
6. Standardizing column names across datasets
7. Merging all products into a single dataframe
8. Final dataframe saved for reuse.

🤖 Models Trained

Multiple algorithms were evaluated:

1. Logistic Regression (with class_weight='balanced')
2. Multinomial Naive Bayes
3. Linear SVM

Class Imbalance Handling

1. Used class_weight='balanced' where applicable
2. Evaluated train vs test accuracy gap
3. Focused on generalization, inference speed, and model size

Hyperparameter Optimization (Optuna)

Used Optuna to tune:
1. TF-IDF parameters
     1. Model-specific hyperparameters
     2. Objective optimized: Validation Accuracy
2. Multiple trials logged automatically

📊 Experiment Tracking (MLflow)

Tracked using MLflow:
1. Parameters
2. Metrics
3. Artifacts
4. Model versions

Logged Metrics:
1. Training Accuracy
2. Test Accuracy
3. Fit Time
4. Test Time
5. Model Size (MB)

Custom Run Names
Each run tagged with: 
     1. Model name
     2. Optimization strategy
     3. Dataset info

Visualization
1. Metric comparison plots
2. Hyperparameter plots (Optuna → MLflow)
3. Model performance comparison across runs

🏆 Best Model Selection

After comparison, Multinomial Naive Bayes was selected as the best model because it:
Achieved competitive accuracy
Had lowest training time
Had fastest inference
Had smallest model size
Generalized well despite class imbalance

🗂️ Model Registry (MLflow)

Best model registered in MLflow Model Registry
Tagged with:
stage: Production
model_type: NaiveBayes
dataset: Flipkart Reviews
Versioned and reproducible

🔄 ML Orchestration (Prefect)

The ML pipeline was converted into a Prefect workflow.

Prefect Tasks:
Load data
Preprocess data
Split train/test
Train model
Evaluate model
Log to MLflow

Flow Features:
Modular task-based design
Re-runnable and fault-tolerant
Cron scheduling supported
Visible in Prefect Dashboard
Prefect server runs locally with UI for monitoring flow runs.

🌐 Backend (Flask)

Flask API to serve the trained MLflow model
Endpoint accepts review text
Returns predicted sentiment
CORS enabled for frontend communication

🎨 Frontend (React)

React app created using create-react-app
UI for entering review text
Displays sentiment prediction
Designed for future extension (product-wise prediction, confidence score)

To Execute:
Open terminal -> cd backend -> run dataprepocessing.py -> run Model_experiment_tracking.py -> ml_orchestration.py
in same terminal -> cd backend -> run app.py
open new terminal -> cd frontend -> npm start
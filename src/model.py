import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_loader import load_stock_data
from src.features import build_features, FEATURE_COLS

def split_data(df, feature_cols, target_col="target", train_ratio=0.8):
    split = int(len(df) * train_ratio)
    X = df[feature_cols]
    y = df[target_col]
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], df.iloc[split:]

def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "LogisticRegression" else X_train.values
        X_te = X_test_s  if name == "LogisticRegression" else X_test.values
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "preds": preds, "proba": proba, "accuracy": acc, "scaler": scaler}
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']:.4f})")
    return results, best_name

def save_model(model, scaler, path="model.joblib", scaler_path="scaler.joblib"):
    joblib.dump(model, path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {path}")

if __name__ == "__main__":
    df_raw = load_stock_data("SPY")
    df = build_features(df_raw)
    X_train, X_test, y_train, y_test, df_test = split_data(df, FEATURE_COLS)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)
    best = results[best_name]
    save_model(best["model"], best["scaler"])

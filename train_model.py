import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ===== إعداد المسارات =====
DATA_PATH = "DARWIN.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_artifacts.pkl")

# ===== الأعمدة (features) المختارة من DARWIN (20 ميزة) =====
FEATURE_NAMES = [
    "total_time23",
    "total_time17",
    "total_time15",
    "air_time17",
    "paper_time23",
    "air_time23",
    "air_time15",
    "total_time13",
    "mean_speed_in_air17",
    "paper_time17",
    "total_time9",
    "total_time6",
    "total_time16",
    "mean_acc_in_air17",
    "gmrt_in_air17",
    "total_time8",
    "mean_gmrt17",
    "air_time22",
    "mean_jerk_in_air17",
    "total_time2",
]


def load_data():
    """تحميل البيانات وتجهيز X, y"""
    df = pd.read_csv(DATA_PATH)

    # نتأكد من وجود العمود class
    if "class" not in df.columns:
        raise ValueError("عمود 'class' غير موجود في ملف DARWIN.csv")

    # نحذف الصفوف اللي ما فيها label
    df = df.dropna(subset=["class"])

    # نجهّز y (H -> 0, P -> 1)
    y = df["class"].map({"H": 0, "P": 1})
    if y.isnull().any():
        raise ValueError("يوجد قيم غير H/P في عمود class")

    # نحذف الأعمدة اللي ما نحتاجها
    drop_cols = []
    if "ID" in df.columns:
        drop_cols.append("ID")
    drop_cols.append("class")

    X = df.drop(columns=drop_cols)

    # نتأكد أن كل الـFEATURE_NAMES موجودة
    missing = [f for f in FEATURE_NAMES if f not in X.columns]
    if missing:
        raise ValueError(f"الأعمدة التالية غير موجودة في الداتا: {missing}")

    # نرجّع فقط الأعمدة المختارة
    X_sel = X[FEATURE_NAMES].copy()

    return X_sel, y


def compute_feature_stats(X_sel):
    """حساب min/max/mean/step لكل feature لاستخدامها في السلايدر."""
    stats = {}
    for col in X_sel.columns:
        col_data = X_sel[col].astype(float)
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        col_mean = float(col_data.mean())

        # لو min==max نعطي شوية مارجن
        if col_min == col_max:
            col_min -= 1.0
            col_max += 1.0

        step = (col_max - col_min) / 100.0 if col_max > col_min else 0.01

        stats[col] = {
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "step": step,
        }
    return stats


def train_and_save():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Pipeline: StandardScaler + RandomForest
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    print("==== Evaluation on hold-out set ====")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

    # نحسب إحصائيات الـfeatures على X_sel
    feature_stats = compute_feature_stats(X)

    # ماب الليبلز للعرض
    label_map = {0: "Healthy", 1: "Patient"}

    # نتأكد مجلد artifacts موجود
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    artifacts = {
        "pipeline": pipe,
        "feature_names": FEATURE_NAMES,
        "feature_stats": feature_stats,
        "label_map": label_map,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Saved model artifacts to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
# trains the pass/fail model for the student pass predictor
# target: pass if G3 >= 10, G2 is left out so it works early in the year
# saves model/model.joblib, model/metrics.json and the charts in assets/
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/raw.csv")
TARGET = "result"
TASK = "classification"
RANDOM_STATE = 42
# column -> name used in the app
FEATURES = {
    "subject": "subject",
    "sex": "sex",
    "age": "age",
    "studytime": "weekly_study_time",      # 1:<2h 2:2-5h 3:5-10h 4:>10h
    "failures": "past_class_failures",
    "absences": "absences",
    "higher": "wants_higher_education",
    "internet": "internet_at_home",
    "schoolsup": "extra_school_support",
    "famsup": "family_support",
    "goout": "going_out_with_friends",      # 1 very low .. 5 very high
    "Walc": "weekend_alcohol_use",          # 1 very low .. 5 very high
    "health": "health_status",              # 1 very bad .. 5 very good
    "Medu": "mother_education",             # 0 none .. 4 higher education
    "G1": "first_period_grade",             # 0..20
}

Path("model").mkdir(exist_ok=True)
Path("assets").mkdir(exist_ok=True)

raw = pd.read_csv(DATA)
df = raw[list(FEATURES)].rename(columns=FEATURES)
df[TARGET] = np.where(raw["G3"] >= 10, "Pass", "Fail")
print("shape:", df.shape)
print("missing:\n", df.isna().sum()[df.isna().sum() > 0])
if TASK == "classification":
    print("target balance:\n", df[TARGET].value_counts(normalize=True))

X, y = df.drop(columns=[TARGET]), df[TARGET]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

if TASK == "classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    binary = y.nunique() == 2
    scoring = "roc_auc" if binary else "f1_macro"
    stratify = y
else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    scoring = "neg_root_mean_squared_error"
    stratify = None

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)

cv_scores = {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    s = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring=scoring)
    cv_scores[name] = float(s.mean())
    print(f"{name:22s} CV {scoring}: {s.mean():.4f} ± {s.std():.4f}")

best_name = max(cv_scores, key=cv_scores.get)
best = Pipeline([("pre", pre), ("model", models[best_name])]).fit(X_tr, y_tr)
pred = best.predict(X_te)

test = {}
if TASK == "classification":
    test["accuracy"] = accuracy_score(y_te, pred)
    test["f1_macro"] = f1_score(y_te, pred, average="macro")
    if binary:
        test["roc_auc"] = roc_auc_score(y_te == best.classes_[1], best.predict_proba(X_te)[:, 1])
    ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Blues")
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout(); plt.savefig("assets/confusion_matrix.png", dpi=150); plt.close()
else:
    test["rmse"] = float(np.sqrt(mean_squared_error(y_te, pred)))
    test["mae"] = mean_absolute_error(y_te, pred)
    test["r2"] = r2_score(y_te, pred)
    plt.scatter(y_te, pred, alpha=0.4)
    lo, hi = float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"Predicted vs actual — {best_name}")
    plt.tight_layout(); plt.savefig("assets/pred_vs_actual.png", dpi=150); plt.close()

imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE)
order = np.argsort(imp.importances_mean)[-10:]
plt.barh(np.array(X.columns)[order], imp.importances_mean[order])
plt.title("Top features (permutation importance)")
plt.tight_layout(); plt.savefig("assets/feature_importance.png", dpi=150); plt.close()

test = {k: round(float(v), 4) for k, v in test.items()}
print("TEST:", best_name, test)

# defaults and options for the app inputs
schema = {}
for c in num_cols:
    schema[c] = {"type": "number", "min": float(X[c].min()), "max": float(X[c].max()),
                 "default": float(X[c].median()), "integer": bool(pd.api.types.is_integer_dtype(X[c]))}
for c in cat_cols:
    schema[c] = {"type": "category", "options": sorted(X[c].dropna().astype(str).unique().tolist()),
                 "default": str(X[c].mode()[0])}

joblib.dump(best, "model/model.joblib")
Path("model/metrics.json").write_text(json.dumps({
    "task": TASK, "target": TARGET, "best_model": best_name, "cv_metric": scoring,
    "cv_scores": cv_scores, "test_metrics": test, "n_rows": int(len(df)),
    "classes": sorted(map(str, y.unique())) if TASK == "classification" else None,
    "schema": schema,
}, indent=2))
print("saved model/model.joblib and model/metrics.json")

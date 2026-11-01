"""Task 1: Student Score Prediction (Linear Regression + bonus)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

DATA_PATH = "data/StudentPerformanceFactors.csv"
OUT_DIR = "outputs"
TARGET = "Exam_Score"
MAIN_FEATURE = "Hours_Studied"
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<35} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return r2


# 1. Load + clean
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)
df = df.drop_duplicates().dropna(subset=[MAIN_FEATURE, TARGET])
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
# Exam scores above 100 are invalid
df = df[df[TARGET] <= 100]
print("Clean shape:", df.shape)

# 2. Basic visualization
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df[MAIN_FEATURE].hist(ax=ax[0], bins=20)
ax[0].set_title("Hours studied")
df[TARGET].hist(ax=ax[1], bins=20)
ax[1].set_title("Exam score")
ax[2].scatter(df[MAIN_FEATURE], df[TARGET], alpha=0.3)
ax[2].set(title="Hours studied vs exam score", xlabel=MAIN_FEATURE, ylabel=TARGET)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/data_overview.png")
plt.close()

# 3. Train/test split + linear regression (study hours only)
X = df[[MAIN_FEATURE]]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("\n--- Linear regression (Hours_Studied) ---")
evaluate("Linear", y_test, pred)
print(f"Coefficient={model.coef_[0]:.3f}  Intercept={model.intercept_:.3f}")

# 4. Visualize predictions
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, alpha=0.3, label="Actual")
order = np.argsort(X_test[MAIN_FEATURE].values)
plt.plot(X_test[MAIN_FEATURE].values[order], pred[order], color="red", label="Predicted")
plt.xlabel(MAIN_FEATURE)
plt.ylabel(TARGET)
plt.title("Linear regression fit")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/linear_fit.png")
plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, alpha=0.3)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/actual_vs_predicted.png")
plt.close()

# 5. BONUS A: polynomial regression vs linear
print("\n--- Bonus: polynomial regression ---")
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, alpha=0.2, label="Actual")
grid = np.linspace(X[MAIN_FEATURE].min(), X[MAIN_FEATURE].max(), 200).reshape(-1, 1)
grid_df = pd.DataFrame(grid, columns=[MAIN_FEATURE])
for degree in (1, 2, 3):
    poly = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X_train, y_train)
    evaluate(f"Polynomial (degree {degree})", y_test, poly.predict(X_test))
    plt.plot(grid, poly.predict(grid_df), label=f"degree {degree}")
plt.xlabel(MAIN_FEATURE)
plt.ylabel(TARGET)
plt.title("Polynomial vs linear")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/polynomial_comparison.png")
plt.close()

# 6. BONUS B: different feature combinations
print("\n--- Bonus: feature combinations ---")
combos = {
    "Hours only": [MAIN_FEATURE],
    "Hours + Sleep": [MAIN_FEATURE, "Sleep_Hours"],
    "Hours + Attendance": [MAIN_FEATURE, "Attendance"],
    "Hours + Previous_Scores": [MAIN_FEATURE, "Previous_Scores"],
    "Hours + Sleep + Attendance + Prev": [MAIN_FEATURE, "Sleep_Hours", "Attendance", "Previous_Scores"],
}
results = {}
for name, cols in combos.items():
    cols = [c for c in cols if c in df.columns]
    Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    results[name] = evaluate(name, yte, m.predict(Xte))

plt.figure(figsize=(8, 4))
plt.barh(list(results.keys()), list(results.values()))
plt.xlabel("R2 score")
plt.title("Feature combination comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_combinations.png")
plt.close()

print(f"\nPlots saved to ./{OUT_DIR}/")

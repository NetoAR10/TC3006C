import numpy as np
import pandas as pd


# =============================
# 1) Funciones del modelo
# =============================
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

def log_loss(y_true, y_pred):
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def gradient_descent_logloss(X, y, theta, alpha, epochs):
    n = y.shape[0]
    for _ in range(epochs):
        p = hypothesis(X, theta) 
        grad = np.dot(X.T, (p - y)) / n 
        theta = theta - alpha * grad
    return theta

def predict_proba(X, theta):
    return hypothesis(X, theta)

def accuracy(y_true, y_hat):
    return np.mean(y_true == y_hat)

def confusion_matrix_binary(y_true, y_hat):
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))
    return tp, fp, fn, tn


# =============================
# 2) Utilidades de preprocesamiento
# =============================
def stratified_train_test_split(X_df, y, test_ratio):
    rng = np.random.default_rng()
    y = np.asarray(y).astype(int)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    n_pos_test = int(len(idx_pos) * test_ratio)
    n_neg_test = int(len(idx_neg) * test_ratio)

    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.concatenate([idx_pos[n_pos_test:], idx_neg[n_neg_test:]])
    rng.shuffle(test_idx); rng.shuffle(train_idx)

    return (X_df.iloc[train_idx].copy(), X_df.iloc[test_idx].copy(),
            y[train_idx].astype(np.float64), y[test_idx].astype(np.float64))

def standardize_train_test(X_train_df, X_test_df, numeric_cols):
    means = X_train_df[numeric_cols].mean()
    stds  = X_train_df[numeric_cols].std(ddof=0).replace(0, 1.0)
    X_train_df[numeric_cols] = (X_train_df[numeric_cols] - means) / stds
    X_test_df[numeric_cols]  = (X_test_df[numeric_cols]  - means) / stds
    return means, stds


# =============================
# 3) Cargar y preparar el dataset
# =============================
CSV_PATH = "diabetes_prediction_dataset.csv"
df = pd.read_csv(CSV_PATH)

target_col = "diabetes"
bin_cols = ["hypertension", "heart_disease"] 
num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"] 

df[bin_cols] = df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)

X_df = pd.concat([df[bin_cols], df[num_cols]], axis=1)
y = df[target_col].astype(np.float64).to_numpy()


# =============================
# 4) Split y estandarización
# =============================
X_train_df, X_test_df, y_train, y_test = stratified_train_test_split(X_df, y, test_ratio=0.6)

means, stds = standardize_train_test(X_train_df, X_test_df, numeric_cols=num_cols)

X_train = np.c_[np.ones(len(X_train_df), dtype=np.float64), X_train_df.to_numpy(dtype=np.float64)]
X_test  = np.c_[np.ones(len(X_test_df),  dtype=np.float64), X_test_df.to_numpy(dtype=np.float64)]


# =============================
# 5) Entrenamiento
# =============================
theta0 = np.zeros(X_train.shape[1], dtype=np.float64)
theta = gradient_descent_logloss(
    X_train, y_train, theta0,
    alpha=0.1, epochs=500
)


# =============================
# 6) Evaluación
# =============================
p_train = predict_proba(X_train, theta)
p_test  = predict_proba(X_test,  theta)

yhat_train = (p_train >= 0.5).astype(int)
yhat_test  = (p_test  >= 0.5).astype(int)

print("\n=== TRAIN ===")
print("LogLoss:", log_loss(y_train, p_train))
print("Accuracy:", accuracy(y_train, yhat_train))
print("Confusion (TP, FP, FN, TN):", confusion_matrix_binary(y_train, yhat_train))

print("\n=== TEST ===")
print("LogLoss:", log_loss(y_test, p_test))
print("Accuracy:", accuracy(y_test, yhat_test))
print("Confusion (TP, FP, FN, TN):", confusion_matrix_binary(y_test, yhat_test))
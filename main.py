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
    # Devuelve en el mismo orden que ya usabas:
    return tp, fp, fn, tn

# === NUEVO: Recall y F1 (binario) ===
def recall_score_binary(y_true, y_hat):
    """TP / (TP + FN)"""
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_hat)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0

def f1_score_binary(y_true, y_hat):
    """2 * (precision * recall) / (precision + recall)"""
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_hat)
    prec_denom = tp + fp
    rec_denom  = tp + fn
    precision = tp / prec_denom if prec_denom > 0 else 0.0
    recall    = tp / rec_denom  if rec_denom  > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


# =============================
# 2) Utilidades de preprocesamiento
# =============================
def stratified_train_val_test_split(X_df, y, val_ratio=0.1, test_ratio=0.1, random_state=None):
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    def split_indices(class_idx):
        n = len(class_idx)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        test_i = class_idx[:n_test]
        val_i  = class_idx[n_test:n_test + n_val]
        train_i = class_idx[n_test + n_val:]
        return train_i, val_i, test_i

    pos_train, pos_val, pos_test = split_indices(idx_pos)
    neg_train, neg_val, neg_test = split_indices(idx_neg)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx   = np.concatenate([pos_val,   neg_val])
    test_idx  = np.concatenate([pos_test,  neg_test])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    return (
        X_df.iloc[train_idx].copy(),
        X_df.iloc[val_idx].copy(),
        X_df.iloc[test_idx].copy(),
        y[train_idx].astype(np.float64),
        y[val_idx].astype(np.float64),
        y[test_idx].astype(np.float64),
    )


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
# 4) Split (80/10/10) y estandarización
# =============================
X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = stratified_train_val_test_split(
    X_df, y, val_ratio=0.10, test_ratio=0.10, random_state=42
)

means = X_train_df[num_cols].mean()
stds  = X_train_df[num_cols].std(ddof=0).replace(0, 1.0)
X_train_df[num_cols] = (X_train_df[num_cols] - means) / stds
X_val_df[num_cols]   = (X_val_df[num_cols]   - means) / stds
X_test_df[num_cols]  = (X_test_df[num_cols]  - means) / stds

X_train = np.c_[np.ones(len(X_train_df), dtype=np.float64), X_train_df.to_numpy(dtype=np.float64)]
X_val   = np.c_[np.ones(len(X_val_df),   dtype=np.float64), X_val_df.to_numpy(dtype=np.float64)]
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
p_val   = predict_proba(X_val,   theta)
p_test  = predict_proba(X_test,  theta)

yhat_train = (p_train >= 0.5).astype(int)
yhat_val   = (p_val   >= 0.5).astype(int)
yhat_test  = (p_test  >= 0.5).astype(int)

print("\n=== TRAIN (80%) ===")
print("LogLoss:", log_loss(y_train, p_train))
print("Accuracy:", accuracy(y_train, yhat_train))
print("Recall:", recall_score_binary(y_train, yhat_train))
print("F1:", f1_score_binary(y_train, yhat_train))
print("Confusion (TP, FP, FN, TN):", confusion_matrix_binary(y_train, yhat_train))

print("\n=== VALIDACIÓN (10%) ===")
print("LogLoss:", log_loss(y_val, p_val))
print("Accuracy:", accuracy(y_val, yhat_val))
print("Recall:", recall_score_binary(y_val, yhat_val))
print("F1:", f1_score_binary(y_val, yhat_val))
print("Confusion (TP, FP, FN, TN):", confusion_matrix_binary(y_val, yhat_val))

print("\n=== TEST (10%) ===")
print("LogLoss:", log_loss(y_test, p_test))
print("Accuracy:", accuracy(y_test, yhat_test))
print("Recall:", recall_score_binary(y_test, yhat_test))
print("F1:", f1_score_binary(y_test, yhat_test))
print("Confusion (TP, FP, FN, TN):", confusion_matrix_binary(y_test, yhat_test))

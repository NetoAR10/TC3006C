import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve,
    average_precision_score, roc_curve, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree as sktree

# =============================
# 0) Helpers
# =============================
def describe_labels(y, name="SET"):
    y = np.asarray(y).astype(int)
    n  = y.size
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_pct = pos / n if n else 0.0
    neg_pct = neg / n if n else 0.0
    print(f"{name}: n={n} | pos={pos} ({pos_pct:.2%}) | neg={neg} ({neg_pct:.2%})")

def resample_hybrid_train(X, y, max_neg_mult=4, random_state=42):
    """
    Híbrido para desbalance binario: conserva TODOS los positivos y toma
    hasta max_neg_mult * (#positivos) negativos sin reemplazo.
    Deja VALID y TEST intactos.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y).astype(int)

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("No se puede re-muestrear: falta alguna clase en TRAIN.")

    target_neg = min(len(neg_idx), max_neg_mult * len(pos_idx))
    neg_sample = rng.choice(neg_idx, size=target_neg, replace=False)

    keep_idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(keep_idx)

    X_res = X.iloc[keep_idx].copy()
    y_res = y_arr[keep_idx]
    return X_res, y_res

# =============================
# 1) Tratamiento de datos
# =============================
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
           'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
data = pd.read_csv('diabetes_prediction_dataset.csv', names=columns, header=0)

def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

data['smoking_history'] = data['smoking_history'].apply(recategorize_smoking)
pd.set_option('future.no_silent_downcasting', True)
data = data[data['gender'] != 'Other']

def perform_one_hot_encoding(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    return df

data = perform_one_hot_encoding(data, 'gender')
data = perform_one_hot_encoding(data, 'smoking_history')

df = data.drop_duplicates()
df_x = df.drop(columns=['diabetes'])
df_y = df['diabetes']

# =============================
# 2) Split 80/10/10 estratificado
# =============================
# Primero 80% train + 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    df_x, df_y, test_size=0.20, random_state=42, stratify=df_y
)
# Luego 50/50 de temp -> 10% valid, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nDistribución ORIGINAL por split (antes de re-muestreo):")
describe_labels(y_train, "TRAIN")
describe_labels(y_val,   "VALID")
describe_labels(y_test,  "TEST")

# =============================
# 3) Re-muestreo híbrido SOLO en TRAIN (1:4 por defecto)
# =============================
X_train_rs, y_train_rs = resample_hybrid_train(X_train, y_train, max_neg_mult=4, random_state=42)

print("\nDistribución TRAS re-muestreo en TRAIN (híbrido 1:4):")
describe_labels(y_train_rs, "TRAIN* (resampled)")
describe_labels(y_val,      "VALID (intacto)")
describe_labels(y_test,     "TEST  (intacto)")

# =============================
# 4) Modelo: Random Forest (entrenar con TRAIN resampleado)
# =============================
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
    bootstrap=True
)
rf_clf.fit(X_train_rs, y_train_rs)

# =============================
# 5) Evaluación
# =============================
# Predicciones y probabilidades
y_pred_tr = rf_clf.predict(X_train_rs)
y_proba_tr = rf_clf.predict_proba(X_train_rs)[:, 1]

y_pred_va = rf_clf.predict(X_val)
y_proba_va = rf_clf.predict_proba(X_val)[:, 1]

y_pred_te = rf_clf.predict(X_test)
y_proba_te = rf_clf.predict_proba(X_test)[:, 1]

print("\n=== TRAIN* (resampled) ===")
print("Accuracy:", accuracy_score(y_train_rs, y_pred_tr))
print("ROC AUC:", roc_auc_score(y_train_rs, y_proba_tr))
print("Log Loss:", log_loss(y_train_rs, y_proba_tr))
print("OOB Score:", rf_clf.oob_score_)
print("Matriz de confusión (TRAIN*):\n", confusion_matrix(y_train_rs, y_pred_tr))
print("Reporte de clasificación (TRAIN*):\n", classification_report(y_train_rs, y_pred_tr))

print("\n=== VALID (10%) ===")
print("Accuracy:", accuracy_score(y_val, y_pred_va))
print("ROC AUC:", roc_auc_score(y_val, y_proba_va))
print("Log Loss:", log_loss(y_val, y_proba_va))
print("Matriz de confusión (VALID):\n", confusion_matrix(y_val, y_pred_va))
print("Reporte de clasificación (VALID):\n", classification_report(y_val, y_pred_va))

print("\n=== TEST (10%) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_te))
print("ROC AUC:", roc_auc_score(y_test, y_proba_te))
print("Log Loss:", log_loss(y_test, y_proba_te))
print("Matriz de confusión (TEST):\n", confusion_matrix(y_test, y_pred_te))
print("Reporte de clasificación (TEST):\n", classification_report(y_test, y_pred_te))

# Visual de matriz de confusión en TEST
ConfusionMatrixDisplay.from_estimator(rf_clf, X_test, y_test)
plt.title("Matriz de confusión (TEST)")
plt.show()

# =============================
# 6) Importancia de características
# =============================
feat_imp = pd.DataFrame({
    "feature": df_x.columns,
    "importance": rf_clf.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=feat_imp.head(15), x="importance", y="feature")
plt.title("Random Forest - Importancia de características (Top 15)")
plt.show()

# =============================
# 7) Curvas ROC y Precision-Recall (TEST)
# =============================
fpr, tpr, _ = roc_curve(y_test, y_proba_te)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (AUC = %.3f)" % roc_auc_score(y_test, y_proba_te))
plt.show()

prec, rec, _ = precision_recall_curve(y_test, y_proba_te)
ap = average_precision_score(y_test, y_proba_te)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall (AP = %.3f)" % ap)
plt.show()


# =============================
# 8) Visualizar un árbol individual del bosque
# =============================
est = rf_clf.estimators_[0]
plt.figure(figsize=(20, 10))
sktree.plot_tree(est, feature_names=df_x.columns, class_names=['No Diabetes', 'Diabetes'],
                 filled=True, max_depth=3)
plt.title("Un árbol dentro del Random Forest (profundidad limitada)")
plt.show()

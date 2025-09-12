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
# 2) Modelo: Random Forest
# =============================
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=42, stratify=df_y
)

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,             # déjalo libre; o usa 8-12 si quieres más sesgo/menos varianza
    min_samples_leaf=3,         # pequeño regularizador para reducir sobreajuste
    class_weight='balanced',    # útil si hay desbalance
    n_jobs=-1,
    random_state=42,
    oob_score=True
)
rf_clf.fit(x_train, y_train)

y_pred = rf_clf.predict(x_test)
y_proba = rf_clf.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Log Loss:", log_loss(y_test, y_proba))
print("OOB Score:", rf_clf.oob_score_)

ConfusionMatrixDisplay.from_estimator(rf_clf, x_test, y_test)
plt.title("Matriz de confusión")
# plt.show()

# =============================
# 3) Importancia de características
# =============================
feat_imp = pd.DataFrame({
    "feature": df_x.columns,
    "importance": rf_clf.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=feat_imp.head(15), x="importance", y="feature")
plt.title("Random Forest - Importancia de características (Top 15)")
# plt.show()

# =============================
# 4) Curvas ROC y Precision-Recall
# =============================
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
# plt.show()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (AP = %.3f)" % ap)
# plt.show()

# =============================
# 5) Validación cruzada estratificada
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(rf_clf, df_x, df_y, cv=cv, scoring='accuracy', n_jobs=-1)
print("Accuracy promedio CV:", cv_acc.mean())
print("Desviación estándar CV:", cv_acc.std())

# =============================
# 6) Visualizar un árbol individual del bosque
# =============================
est = rf_clf.estimators_[0]
plt.figure(figsize=(20, 10))
sktree.plot_tree(est, feature_names=df_x.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, max_depth=3)
plt.title("Un árbol dentro del Random Forest (profundidad limitada)")
#plt.show()

# =============================
# 7) Predicción con datos de ejemplo (mismas transformaciones)
# =============================
example_data = pd.DataFrame({
    "gender": ["Male"],
    "age": [50],
    "hypertension": [1],
    "heart_disease": [1],
    "smoking_history": ["current"],
    "bmi": [30.0],
    "HbA1c_level": [8.7],
    "blood_glucose_level": [120]
})
example_data["smoking_history"] = example_data["smoking_history"].apply(recategorize_smoking)
example_data = perform_one_hot_encoding(example_data, 'gender')
example_data = perform_one_hot_encoding(example_data, 'smoking_history')
example_data = example_data.reindex(columns=df_x.columns, fill_value=0)

example_pred = rf_clf.predict(example_data)
print("Predicción para ejemplo:", "Diabetes" if example_pred[0] == 1 else "No Diabetes")


# Predicciones train
y_pred_tr = rf_clf.predict(x_train)
y_proba_tr = rf_clf.predict_proba(x_train)[:, 1]

# Predicciones test (si aún no las tienes)
y_pred_te = rf_clf.predict(x_test)
y_proba_te = rf_clf.predict_proba(x_test)[:, 1]

print("TRAIN  Accuracy:", accuracy_score(y_train, y_pred_tr))
print("TRAIN  AUC     :", roc_auc_score(y_train, y_proba_tr))
print("TRAIN  LogLoss :", log_loss(y_train, y_proba_tr))

print("TEST   Accuracy:", accuracy_score(y_test, y_pred_te))
print("TEST   AUC     :", roc_auc_score(y_test, y_proba_te))
print("TEST   LogLoss :", log_loss(y_test, y_proba_te))

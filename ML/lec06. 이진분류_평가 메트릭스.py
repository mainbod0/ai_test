import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve

from sklearn.preprocessing import Binarizer

#-------------------- 차트 관련 속성 (한글처리, 그리드) -----------
plt.rcParams['font.family']= 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False



df = pd.read_csv('C:\\AI\pythonProject\\venv\\datasets\\diabetes.csv')
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 768 entries, 0 to 767
# Data columns (total 9 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Pregnancies               768 non-null    int64
#  1   Glucose                   768 non-null    int64
#  2   BloodPressure             768 non-null    int64
#  3   SkinThickness             768 non-null    int64
#  4   Insulin                   768 non-null    int64
#  5   BMI                       768 non-null    float64
#  6   DiabetesPedigreeFunction  768 non-null    float64
#  7   Age                       768 non-null    int64
#  8   Outcome                   768 non-null    int64
# dtypes: float64(2), int64(7)
# memory usage: 54.1 KB
df.shape
# (768, 9)
df.rename(columns={'Outcome':'target'}, inplace=True)

""" Sampling(N개) """
df[df['target']==1].sample(n=3, random_state=21, ignore_index=True)
#    Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  target
# 0           10      115              0  ...                     0.261   30       1
# 1            0      198             66  ...                     0.502   28       1
# 2           10      125             70  ...                     0.205   41       1
# [3 rows x 9 columns]

""" Sampling(N%) """
dfs = df[df['target']==1].sample(frac=0.1, random_state=55, ignore_index=True)
#      Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  target
# 0             3      173             84  ...                     0.258   22       1
# 1             3      162             52  ...                     0.652   24       1
# 2             1      173             74  ...                     0.088   38       1
# ........
# 25            0      179             90  ...                     0.686   23       1
# 26            0      119              0  ...                     0.141   24       1
# [27 rows x 9 columns]

df['target'].value_counts()
# target
# 0    500
# 1    268
# Name: count, dtype: int64

""" EDA """
df.hist(figsize=(8,6))
plt.tight_layout()
plt.show()

sns.pairplot(data=df, hue='target')
plt.show()

""" train, test 분리 """
y = df['target']
X = df.drop(['target'], axis=1)
X.shape
# (768, 8)
y.shape
# (768,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# (614, 8), (154, 8), (614,), (154,)

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)
df_var = model.decision_function(X_test)

print(f"y_test : {y_test[:5]}")
# y_test :
# 665    0
# 144    0
# 662    1
# 719    1
# 745    0
# Name: target, dtype: int64
print(f"pred : {pred[:5]}")
# pred : [0 0 1 0 0]
print(f"proba : {proba[:5]}")
# proba :
# [[0.8915017  0.1084983 ]
# [0.51374659 0.48625341]
# [0.27521522 0.72478478]
# [0.76299872 0.23700128]
# [0.65757233 0.34242767]]
print(f"dec.func : {df_var[:5]}")
# dec.func : [-2.10617284 -0.05500022  0.96832132 -1.16919083 -0.6524943 ]

accuracy = accuracy_score(y_test, pred)
# 0.7532467532467533

""" DecisionTreeClassifier : feature importances_ """
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
pred =  model.predict(X_test)
proba = model.predict_proba(X_test)

model.feature_importances_
# array([0.05803448, 0.30822021, 0.10151324, 0.04381545, 0.0322802 ,
#        0.15592421, 0.15471815, 0.14549406])

plt.figure(figsize=(4,2))
s = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
sns.barplot(x=s.values, y=s.index)
plt.show()

""" LogisticRegression : coef_ """
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
df_var = model.decision_function(X_test)

model.coef_
# array([[ 0.17570762,  0.03666465, -0.01825319, -0.00260341, -0.00119442,
#          0.10459711,  0.97007514,  0.00952585]])
model.coef_.reshape(-1)
# array([ 0.17570762,  0.03666465, -0.01825319, -0.00260341, -0.00119442,
#         0.10459711,  0.97007514,  0.00952585])
plt.figure(figsize=(4,2))
s = pd.Series(model.coef_.reshape(-1), index=X_train.columns).sort_values()
sns.barplot(x=s.values, y=s.index)
plt.show()

confusion_matrix(y_test, pred)
# array([[87, 13],
#        [25, 29]], dtype=int64)

classification_report(y_test, pred)
#               precision    recall  f1-score   support
#            0       0.78      0.87      0.82       100
#            1       0.69      0.54      0.60        54
#     accuracy                           0.75       154
#    macro avg       0.73      0.70      0.71       154
# weighted avg       0.75      0.75      0.74       154


proba[:5]
# array([[0.8915017 , 0.1084983 ],
#        [0.51374659, 0.48625341],
#        [0.27521522, 0.72478478],
#        [0.76299872, 0.23700128],
#        [0.65757233, 0.34242767]])
proba_c1 = proba[:, 1]

precision, recall, th = precision_recall_curve(y_test, proba_c1)
plt.plot(th, precision[:len(th)], label='precision')
plt.plot(th, recall[:len(th)], label='recall')
plt.legend()
plt.title('Precision-Recall Curv')
plt.show()


""" roc_auc_score """

auc_score = roc_auc_score(y_test, proba_c1)
print(f"AUC 점수 : {auc_score:.5f}")
# AUC 점수 : 0.78593

cr = confusion_matrix(y_test, pred)

TN = cr[0][0]
FN = cr[1][0]
FP = cr[0][1]
TP = cr[1][1]

FPR = FP / (FP + TN)
# 0.13
TPR_recall = TP / (FN + TP)
# 0.5370370370370371

print(classification_report(y_test, pred))
#               precision    recall  f1-score   support
#            0       0.78      0.87      0.82       100
#            1       0.69      0.54      0.60        54
#     accuracy                           0.75       154
#    macro avg       0.73      0.70      0.71       154
# weighted avg       0.75      0.75      0.74       154

fpr, tpr, th = roc_curve(y_test, proba_c1)
fpr.shape
# (44,)
tpr.shape
# (44,)
th.shape
# (44,)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], "--")
plt.plot(FPR, TPR_recall, 'r^')

plt.title("ROC curv")
plt.xlabel("FPR (1-특이도specificity)")
plt.ylabel("TPR = 민감도sensitivity = recall")
plt.title(f"ROC Curv   FPR: {FPR:.4f}, TPR:{TPR_recall:.4f}")
plt.show()

""" decision_function() """
df_var = model.decision_function(X_test)
df_var.shape
# (154,)
df_var.max().round(4)
# 3.4292
df_var.min().round(4)
# -4.4137
df_var.mean().round(4)
# -0.9305
fpr, tpr, th = roc_curve(y_test, df_var)

plt.plot(fpr, tpr, 'o-', label="model")
plt.plot([0, 1], [0, 1], '--', label="th")
plt.plot(FPR, TPR_recall, "r^")

plt.title("ROC curv")
plt.xlabel("FPR (1-특이도specificity)")
plt.ylabel("TPR = 민감도sensitivity = recall")
plt.title(f"FPR: {FPR:.4f}, TPR:{TPR_recall:.4f}")
plt.show()

""" predict_proba() """
proba = model.predict_proba(X_test)

plt.plot(fpr, tpr, 'o-', label="model")
plt.plot([0, 1], [0, 1], '--', label="th")
plt.plot(FPR, TPR_recall, "r^")

plt.title("ROC curv")
plt.xlabel("FPR (1-특이도specificity)")
plt.ylabel("TPR = 민감도sensitivity = recall")
plt.title(f"FPR: {FPR:.4f}, TPR:{TPR_recall:.4f}")
plt.show()









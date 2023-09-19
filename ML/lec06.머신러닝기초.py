import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import warnings
warnings.filterwarnings(action='ignore')


dict = load_iris()
type(dict)
# sklearn.utils._bunch.Bunch
dict.keys()
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

df = pd.DataFrame(data=dict['data'], columns=dict['feature_names'])
df['target'] = dict['target']
df.columns = ['sl', 'sw', 'pl', 'pw', 'target']
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   sl      150 non-null    float64
#  1   sw      150 non-null    float64
#  2   pl      150 non-null    float64
#  3   pw      150 non-null    float64
#  4   target  150 non-null    int32
# dtypes: float64(4), int32(1)
df.shape
# (150, 5)

df.hist()
plt.tight_layout()
plt.show()

""" train, test 분리 """
y = df['target']
x = df.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1211)
# X_train.shape
# X_test.shape
# y_train.shape
# y_test.shape

""" DecisionTreeClassifier """
# class sklearn.tree.DecisionTreeClassifier(*,
# random_state=None,
# #--------------------- hyper-parameter
# criterion='gini',
# max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None,
# #---------------------
# splitter='best',
# min_weight_fraction_leaf=0.0, max_features=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

model = DecisionTreeClassifier(random_state=111, min_samples_split=2)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy_score(y_test, pred)
# 0.9666666666666667

check_df = pd.DataFrame(columns=['y_test', 'pred'])
check_df['y_test'] = y_test
check_df['pred'] = pred
check_df[check_df['y_test'] != check_df['pred']]
#      y_test  pred
# 106       2     1

confusion_matrix(y_test, pred)
# array([[ 9,  0,  0],
#        [ 0, 13,  0],
#        [ 0,  1,  7]], dtype=int64)

print(classification_report(y_test, pred))
#               precision    recall  f1-score   support
#            0       1.00      1.00      1.00         9
#            1       0.93      1.00      0.96        13
#            2       1.00      0.88      0.93         8
#     accuracy                           0.97        30
#    macro avg       0.98      0.96      0.97        30
# weighted avg       0.97      0.97      0.97        30

precision_score(y_test, pred, average="macro")
















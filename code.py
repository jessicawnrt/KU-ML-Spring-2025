## Decision Tree Classifier ##
### *) test size = 0.2 ###

from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

import pandas as pd

iris = load_iris()

iris_data = iris.data
iris_label = iris.target

dt_clf = DecisionTreeClassifier( )

X_train, X_test, y_train, y_test= train_test_split(iris.data, iris.target, test_size=0.2, random_state=2025)

dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))


my_iris = [[5.5, 6.2, 3.6, 2]]
my_pred = dt_clf.predict(my_iris)
my_pred
iris.target_names[my_pred]


## K-Fold ##
### *) n = 5 ###


from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=2025)

kfold = KFold(n_splits=5)
cv_accuracy = []
print('Iris 데이터 세트 크기:',features.shape[0])

n_iter = 0

for train_index, test_index in kfold.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    
    cv_accuracy.append(accuracy)
    
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy)) 


my_iris_kfold = [[5.5, 6.2, 3.6, 2]]
my_pred_kfold = dt_clf.predict(my_iris_kfold)
my_pred_kfold
iris.target_names[my_pred_kfold]


## Stratified K-Fold ##
### *) n = 5 ###

from sklearn.model_selection import StratifiedKFold
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()


dt_clf = DecisionTreeClassifier(random_state=2025)

skfold = StratifiedKFold(n_splits=5)
n_iter=0
cv_accuracy=[]

for train_index, test_index  in skfold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)

    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy)) 

my_iris_s_kfold = [[5.5, 6.2, 3.6, 2]]
my_pred_s_kfold = dt_clf.predict(my_iris_s_kfold)
my_pred_s_kfold
iris.target_names[my_pred_s_kfold]



# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 14:42:54 2020

@author: 이형기
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt   # 그래프 출력시
import matplotlib as mpl    # 그래프 옵션
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score



# 그래프에서 음수 값이 나올 때, 깨지는 현상 방지
mpl.rc('axes',unicode_minus=False)

data=pd.read_csv('Data.csv',encoding='euc-kr')

# 데이터 정보 확인
# val6이 문자타입임을 확인
#data.info()

#인덱스를 확인하여 데이터가 제대로 불러와졌는지 확인
#print(data.columns)  

#결측치 확인
#data.isna().sum()

# val6의 값 확인
#data.val6.value_counts()

# val6의 값들을 숫자형태로 라벨링
data['val6'] = data['val6'].replace('A', 0)
data['val6'] = data['val6'].replace('B', 1)
data['val6'] = data['val6'].replace('C', 2)
data['val6'] = data['val6'].replace('D', 3)
data['val6'] = data['val6'].replace('E', 4)


# 상관행렬
corr = data.drop(['result'],axis=1)
corr = corr.corr()

cor_array = corr.values

#상관관계가 너무큰 속성찾아서 제거

for i in range(0, 31):
    for j in range(i+1, 31):
        if cor_array[i][j] < 1.0 and cor_array[i][j] > 0.9:
            data.drop([i], axis=1)     
corr = data.corr()
    
#상관관계 히트맵
plt.figure(1)

plt.figure(figsize = (15,15))
sns.heatmap(
    corr, square=True,
    vmin = -1, vmax = 1, center = 0, 
    cmap = sns.diverging_palette(220, 20, as_cmap=True)    
)
plt.xticks(rotation=45)



#양의 상관 관계가 강할수록 붉은색,음의 상관관계가 강할수록 파란색


# 결과값 분포 비율 확인
# 1이 0에 비해 압도적으로 많음
# imbalalanced data 이므로 F1-score를 평가지표로 삼음
plt.figure(2)

plt.figure(figsize =(5,10))
sns.countplot(data['result'])
data['result'].value_counts()


# X(데이터), Y(결과)로 구분
# 학습셋:테스트셋 비율을 7:3으로 랜덤하게 분할
X=data.drop('result',axis = 1)
Y=data['result']

X_train,X_test,Y_train,Y_test = \
   train_test_split(X,Y,test_size=0.7,random_state=random.randint(1, 1000))

# 학습셋과 테스트셋의 구성비
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)


# XGBOOST

xgb=XGBClassifier()
#대입
xgb.fit(X_train,Y_train) 
Y_predict=xgb.predict(X_test)

# XGBOOST가 예측한값의 정확도
print('Accuracy : ',accuracy_score(Y_test,Y_predict))
print('F1_score : ',f1_score(Y_test,Y_predict))


#실제값중 얼마나 맞췄는지 나타내는 confusion matrix
plt.figure(3)

plt.figure(figsize = (10 ,10))
sns.heatmap(confusion_matrix(Y_test,Y_predict), 
            annot = True, fmt = "d", linewidths = .5, 
            square = True, 
            cmap = 'PuBuGn');
plt.yticks(rotation = 0)
plt.ylabel('Actual value', size = 10);
plt.xlabel('Predicted value');
plt.title('Confusion Matrix (XGBOOST)\n', size = 20)



# 테스트셋이 편향될 수 있으므로 5-fold cross validation 실시
# f1_score의 평균을 구함
kf = KFold(n_splits = 5)
xgb_score = []

x = X_train.values

y = Y_train.values
model = XGBClassifier()

n=0

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index],x[test_index]
    y_train, y_test = y[train_index],y[test_index]
    
    model.fit(x_train,y_train)
    y_predict_train=model.predict(x_train)
    y_predict_test=model.predict(x_test)
    
    n+= 1
    
    train_accuracy = f1_score(y_train,y_predict_train)
    test_accuracy = f1_score(y_test,y_predict_test)
    xgb_score.append(test_accuracy)
    
    print("XGBOOST cross validation ",n,"time")
    print('train_f1_score : ',train_accuracy)
    print('test_f1_score : ',test_accuracy)
    print("\n")
print('-------------------------------------\n')    
print('avarage xgb_f1_score : ',np.mean(xgb_score), '\n\n')

# Decision Tree

dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt_predict=dt.predict(X_test)

# DT가 예측한값의 정확도
print('Accuracy : ',accuracy_score(Y_test,dt_predict))
print('F1_score : ',f1_score(Y_test,dt_predict))

# DT confusion matrix
plt.figure(4)
plt.figure(figsize = (10, 10))
sns.heatmap(confusion_matrix(Y_test,dt_predict), 
            annot=True, fmt = "d", linewidths = .5, 
            square = True, 
            cmap = 'PuBuGn');
plt.yticks(rotation = 0)
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix (Decision Tree)\n', size = 20)

# 테스트셋이 편향될 수 있으므로 5-fold cross validation 실시
# f1_score의 평균을 구함
kf=KFold(n_splits = 5)
dt_score = []

x = X_train.values
y = Y_train.values
model=DecisionTreeClassifier()

n = 0
for train_index, test_index in kf.split(x):
    x_train, x_test=x[train_index],x[test_index]
    y_train, y_test=y[train_index],y[test_index]
    
    model.fit(x_train,y_train)
    y_predict_train=model.predict(x_train)
    y_predict_test=model.predict(x_test)
    
    n+=1
    
    train_accuracy = f1_score(y_train,y_predict_train)
    test_accuracy = f1_score(y_test,y_predict_test)
    dt_score.append(test_accuracy)
    
    print("Decision tree cross validation ", n, "time")
    print('train_f1_score: ',train_accuracy)
    print('test_f1_score: ',test_accuracy)
    print('\n')

print('-------------------------------------\n')    
print('avarage dt_f1_score: ',np.mean(dt_score), '\n\n')



rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
rf_predict=rf.predict(X_test)

# RF가 예측한값의 정확도
print('Accuracy : ', accuracy_score(Y_test, rf_predict))
print('F1_score : ', f1_score(Y_test, rf_predict))

# RF confusion matrix
plt.figure(5)
plt.figure(figsize = (10, 10))
sns.heatmap(confusion_matrix(Y_test,rf_predict), 
            annot=True, fmt = "d", linewidths = .5, 
            square = True, 
            cmap = 'PuBuGn');
plt.yticks(rotation = 0)
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix (Random Forest)\n', size = 20)



# 테스트셋이 편향될 수 있으므로 5-fold cross validation 실시
# f1_score의 평균을 구함
kf = KFold(n_splits = 5)
rf_score = []

x = X_train.values
y = Y_train.values
model = RandomForestClassifier()

n = 0
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index],x[test_index]
    y_train, y_test = y[train_index],y[test_index]
    
    model.fit(x_train,y_train)
    y_predict_train = model.predict(x_train)
    y_predict_test = model.predict(x_test)
    
    n+=1
    
    train_accuracy = f1_score(y_train,y_predict_train)
    test_accuracy = f1_score(y_test,y_predict_test)
    rf_score.append(test_accuracy)
    
    print("Random Forest cross validation ",n,"time")
    print('train_f1_score: ', train_accuracy)
    print('test_f1_score: ', test_accuracy)
    print('\n')
    
print('-------------------------------------\n')    
print('avarage rf_f1_score: ',np.mean(rf_score), '\n\n')


print('avarage xgb_f1_score : ',np.mean(xgb_score))
print('avarage dt_f1_score: ',np.mean(dt_score))
print('avarage rf_f1_score: ',np.mean(rf_score))


# XGBOOST의 f1_score가 가장 높으므로 최적의 모델임을 확인
# 예측하는데 중요하게 작용한 변수들을 변수 중요도와 함께 출력
xgb=XGBClassifier()
xgb.fit(X_train,Y_train)

plt.figure(6)
plt.figure(figsize=(10,15))
plot_importance(xgb,ax=plt.gca())

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
import tkinter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from xgboost import plot_importance
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from tkinter import filedialog


# 그래프에서 음수 값이 나올 때, 깨지는 현상 방지
mpl.rc('axes',unicode_minus=False)


#tkinter 창으로 파일을 선택
root = tkinter.Tk()
root.withdraw()
dir_path = filedialog.askopenfilename(parent=root,title='Please select a Data.csv',filetypes=(('csv files','*.csv'),('all files','*.*')))
print("\ndir_path : ", dir_path)

data=pd.read_csv(dir_path,encoding='euc-kr')


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


# X(데이터), Y(결과)로 구분
# 학습셋:테스트셋 비율을 7:3으로 랜덤하게 분할
X=data.drop('result',axis = 1)
Y=data['result']

X_train,X_test,Y_train,Y_test = \
   train_test_split(X,Y,test_size=0.3,random_state=random.randint(1, 1000))

# 학습셋과 테스트셋의 구성비
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)

# XGBOOST

xgb=XGBClassifier()
#대입
xgb.fit(X_train,Y_train) 
Y_predict=xgb.predict(X_test)

#print(X_test)

#print(pre_result)
pre_result = pd.DataFrame(Y_test,columns=['result'])
pre_result['predict']=Y_predict
pre_result.to_csv("predict_result.csv")


# XGBOOST가 예측한값의 정확도
print('Accuracy : ',accuracy_score(Y_test,Y_predict))
print('F1_score : ',f1_score(Y_test,Y_predict))


#실제값중 얼마나 맞췄는지 나타내는 confusion matrix
plt.figure(figsize = (10 ,10))
sns.heatmap(confusion_matrix(Y_test,Y_predict), 
            annot = True, fmt = "d", linewidths = .5, 
            square = True, 
            cmap = 'PuBuGn');
plt.yticks(rotation = 0)
plt.ylabel('Actual value', size = 10);
plt.xlabel('Predicted value');
plt.title('Confusion Matrix (XGBOOST)\n', size = 20)
plt.savefig("Confusion Matrix.png")

# XGBOOST의 f1_score가 가장 높으므로 최적의 모델임을 확인
# 예측하는데 중요하게 작용한 변수들을 변수 중요도와 함께 출력
plt.figure(figsize=(10,15))
best=XGBClassifier()
best.fit(X_train,Y_train)
plot_importance(best,ax=plt.gca()) 
plt.title('F1_score : %-10.4f\nAccuracy : %-10.4f\n\nFeature importance' %(np.mean(f1_score(Y_test,Y_predict)), accuracy_score(Y_test,Y_predict)))
plt.savefig("Feature importance.png")

plt.show()

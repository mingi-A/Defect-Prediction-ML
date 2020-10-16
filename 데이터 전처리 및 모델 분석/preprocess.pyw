# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as stats
import tkinter

from tkinter import filedialog

#tkinter 창으로 파일을 선택
root = tkinter.Tk()
root.withdraw()
dir_path = filedialog.askopenfilename(parent=root,title='Please select a Dataset.csv',filetypes=(('csv files','*.csv'),('all files','*.*')))
print("\ndir_path : ", dir_path)

data=pd.read_csv(dir_path,encoding='euc-kr')


# 데이터 분석을 위해 id값 제외 후 불량 판단 결과를 앞으로 가져와서 재정렬
data.drop(['TRACE_KEY','itm_cd', 'itm_location','in_date', 'KEY_TIMESTAMP'], axis='columns', inplace=True)
df = data.reindex(columns=['ProdJud']+list([a for a in data.columns if a != 'ProdJud']))


# 데이터 보안을 위해 colums명을 지정
sec_columns = ['val'+i for i in list(map(str, range(df.shape[1]))) ]
df.columns = sec_columns

df.rename(columns={'val0':'result'}).to_csv("Data.csv",index=None)


# 데이터 정규화
df['val6']=df['val6'].replace(['A','B','C','D','E'],[1,2,3,4,5])
normalization_df = (df - df.mean())/df.std()
normalization_df = normalization_df.dropna(axis=1)


# pearson 상관관계 분석
pearson_val = pd.DataFrame(index=range(0,df.shape[1]), columns=sec_columns)
for i in range(0,df.shape[1]):
    for j in range(0,df.shape[1]):
        corr = stats.pearsonr(normalization_df['val{}'.format(i)], normalization_df['val{}'.format(j)])
        if (i<=j):
            pearson_val['val{}'.format(i)][j] = corr[0]
pearson_val = pearson_val.fillna('')

pearson_val.to_csv("Pearson_val.csv")



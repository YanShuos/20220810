# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:42:29 2022

@author: shuoy
"""


import pandas as pd # 导入Pandas数据处理工具包

df_ads = pd.read_csv('易速鲜花微信软文.csv') # 读入数据
df_ads.head() # 显示前几行数据


#导入数据可视化所需要的库
import matplotlib.pyplot as plt # Matplotlib – Python画图工具库
import seaborn as sns # Seaborn – 统计学数据可视化工具库



plt.plot(df_ads['点赞数'],df_ads['浏览量'],'r.', label='Training data') # 用matplotlib.pyplot的plot方法显示散点图
plt.xlabel('点赞数') # x轴Label
plt.ylabel('浏览量') # y轴Label
plt.legend() # 显示图例
plt.show() # 显示绘图结果！


data = pd.concat([df_ads['浏览量'], df_ads['热度指数']], axis=1) # 浏览量和热度指数
fig = sns.boxplot(x='热度指数', y="浏览量", data=data) # 用seaborn的箱线图画图
fig.axis(ymin=0, ymax=800000); #设定y轴坐标


df_ads.isna().sum() # NaN出现的次数

df_ads = df_ads.dropna() # 把出现了NaN的数据行删掉

X = df_ads.drop(['浏览量'],axis=1) # 特征集，Drop掉标签相关字段
y = df_ads.浏览量 # 标签集

X.head() # 显示前几行数据
y.head() #显示前几行数据

#将数据集进行80%（训练集）和20%（验证集）的分割
from sklearn.model_selection import train_test_split #导入train_test_split工具
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.2, random_state=0)

#%%


from sklearn.linear_model import LinearRegression # 导入线性回归算法模型
linereg_model = LinearRegression() # 使用线性回归算法创建模型

linereg_model.fit(X_train, y_train) # 用训练集数据，训练机器，拟合函数，确定内部参数

y_pred = linereg_model.predict(X_test) #预测测试集的Y值
df_ads_pred = X_test.copy() #测试集特征数据
df_ads_pred['浏览量真值'] = y_test #测试集标签真值
df_ads_pred['浏览量预测值'] = y_pred #测试集标签预测值
df_ads_pred #显示数据

print("线性回归预测集评分：", linereg_model.score(X_test, y_test)) #评估模型
print("线性回归训练集评分：", linereg_model.score(X_train, y_train)) #训练集评分




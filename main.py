import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
df=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')

fit_cols = ['YearBuilt','1stFlrSF','LotArea','2ndFlrSF','BedroomAbvGr','YrSold']
df = df.dropna(subset=fit_cols)
print(df.head())
print(df.shape)


X = df[fit_cols]
y = df.SalePrice
linreg.fit(X, y)

y_pred2 = linreg.predict(test[fit_cols])
test['SalePrice']=y_pred2
print(test.head())
result=test[['Id',"SalePrice"]]
result.to_csv("result.csv",index=False)
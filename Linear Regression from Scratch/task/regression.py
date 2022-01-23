# write your code here
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        X1 = np.array(X)
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if self.fit_intercept:
            X1 = np.hstack((np.ones((X1.shape[0], 1)), X1))
        #print(X1)
        tmp = np.linalg.inv(X1.T.dot(X1))
        beta = tmp.dot(X1.T.dot(y))
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.intercept = 0
            self.coefficient = beta
        return beta

    def predict(self, X):
        return self.intercept + np.array(X).dot(self.coefficient)

    def r2_score(self, y, yhat):
        ybar = np.ones(y.shape) * y.mean()
        err = y - yhat
        dev = y - ybar
        return 1 - err.dot(err) / dev.dot(dev)

    def rmse(self, y, yhat):
        err = y - yhat
        n = y.shape[0]
        mse = err.dot(err) / n
        return np.sqrt(mse)

    def coeff_dict(self):
        return {'Intercept': self.intercept,
                'Coefficient': self.coefficient}

df = pd.read_csv('data_stage4.csv')
#print(df)
reg_custom = CustomLinearRegression(fit_intercept=True)
reg_custom.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred_custom = reg_custom.predict(df[['f1', 'f2', 'f3']])
dict1 = reg_custom.coeff_dict()
dict1['R2'] = reg_custom.r2_score(df['y'], y_pred_custom)
dict1['RMSE'] = reg_custom.rmse(df['y'], y_pred_custom)
#print(dict1)

reg_sci = LinearRegression(fit_intercept=True)
reg_sci.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred_sci = reg_sci.predict(df[['f1', 'f2', 'f3']])
dict2 = {'Intercept': reg_sci.intercept_,
         'Coefficient': reg_sci.coef_,
         'R2': r2_score(df['y'], y_pred_custom),
         'RMSE': np.sqrt(mean_squared_error(df['y'], y_pred_custom))}
dict3 = {}
for k in dict2.keys():
    if k in dict1.keys():
        dict3[k] = dict2[k] - dict1[k]
print(dict3)


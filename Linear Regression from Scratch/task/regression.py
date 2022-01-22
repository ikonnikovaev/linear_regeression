# write your code here
import numpy as np
import pandas as pd

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


data = np.column_stack((np.array([0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9]),
                       np.array([11, 11, 9, 8, 7, 7, 6, 5, 5, 4]),
                       np.array([21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69])))
df = pd.DataFrame(data, columns=['Capacity', 'Age', 'Cost/ton'])

reg_custom = CustomLinearRegression(fit_intercept=True)
reg_custom.fit(df[['Capacity', 'Age']], df['Cost/ton'])
y_pred = reg_custom.predict(df[['Capacity', 'Age']])
dict = reg_custom.coeff_dict()
dict['R2'] = reg_custom.r2_score(df['Cost/ton'], y_pred)
dict['RMSE'] = reg_custom.rmse(df['Cost/ton'], y_pred)
print(dict)



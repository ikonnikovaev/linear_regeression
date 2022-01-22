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

    def coeff_dict(self):
        return {'Intercept': self.intercept,
                'Coefficient': self.coefficient}

data = np.row_stack((np.arange(4.0, 7.5, 0.5),
                     np.array([33, 42, 45, 51, 53, 61, 62])))
df = pd.DataFrame(data.T, columns=['x', 'y'])
#print(df)
X = df['x']
y = np.array(df['y'])
regressor = CustomLinearRegression()
regressor.fit(X, y)
#print(regressor.coeff_dict())


data = np.row_stack((np.arange(4.0, 7.5, 0.5),
                     np.array([1, -3, 2, 5, 0, 3, 6]),
                     np.array([11, 15, 12, 9, 18, 13, 16]),
                     np.array([33, 42, 45, 51, 53, 61, 62])))
df = pd.DataFrame(data.T, columns=['x', 'w', 'z', 'y'])
#print(df)
reg_custom = CustomLinearRegression(fit_intercept=False)
reg_custom.fit(df[['x', 'w', 'z']], df['y'])
y_pred = reg_custom.predict(df[['x', 'w', 'z']])
print(y_pred)

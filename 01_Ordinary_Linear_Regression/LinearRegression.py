import numpy as np

class LinearRegression:

    def fit(self, X, y, intercept=False):

        # record data and dimensions
        if intercept == False:
            ones = np.ones(len(X)).reshape(len(X), 1) # add intercept (if not already included)
            X = np.concatenate((ones, X), axis=1)
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape # number of rows, number of columns - .shape returns a tupple (number_of_rows, number_of_columns).

        # estimate parameters
        XtX = np.dot(self.X.T, self.X)
        XtX_inverse = np.linalg.inv(XtX)
        Xty = np.dot(self.X.T, self.y) # this step measures the correlation between each feature and the target variable, it aggregates how much y tends to increase or decrease as each feature x increases
        self.beta_hats = np.dot(XtX_inverse, Xty) # this scales the raw correlations from the previous step - it adjusts for the fact that features might be correlated with each other

        # make in-sample predictions
        self.y_hat = np.dot(self.X, self.beta_hats) # this line calculates the model's "best guess" for the training data

        # calculate loss
        self.L = .5 * np.sum((self.y - self.y_hat)**2) # sum of squared errors (SSE)

    def predict(self, X_test, intercept = True):

        # form predictions
        self.y_test_hat = np.dot(X_test, self.beta_hats)
import pandas as pd
import numpy as np
import sklearn
from matplotlib import style
from sklearn import linear_model
# from sklearn.util import shuffle
import matplotlib.pyplot as pyplot
import pickle

"""
predict final grades using a linear regression
[first test grade, second test grade, study time, num of past class failures, class absences] --> final grade
"""

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head)

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # data set

predict = "G3"  # Label => what you are looking for

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for var in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)  # make line of best fit LIN ALG
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc

        with open("studentmodel.pickle", "wb") as f:  # saves trained regression model
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Coefficient: ", linear.coef_)
print("Y-Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):  # loops through all the test cases
    print(predictions[i], x_test[i], y_test[i])  # prints: prediction, inputs, actual final grade


# graphs G1 to final grades on a scatter plot
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()




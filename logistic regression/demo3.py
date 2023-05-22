import numpy as np
from sklearn.linear_model import LogisticRegression


x = np.array([[1,1],
              [0,0],
              [1,0],
              [0,1]
              ])
y= np.array([0,0,1,0])

model = LogisticRegression()
model.fit(x , y)
x_test = np.array([[1,1],
                   [1,2],
                   [2,1]])
print(model.predict(x_test))
print(model.get_params())
# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Initialize m,c and learning rate α.

2.Predict y=mx+c.

3.Compute error and gradients.

4.Update 𝑚,𝑐 until convergence.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("50_Startups.csv")
data = data.iloc[:, [0, 4]]
data.columns = ["Population", "Profit"]

data["Population"] = (data["Population"] - data["Population"].mean()) / data["Population"].std()

plt.scatter(data["Population"], data["Profit"])
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()

def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    return (1/(2*m)) * np.sum((h - y)**2)

m = len(data)
X_raw = data["Population"].values.reshape(m, 1)
X = np.append(np.ones((m, 1)), X_raw, axis=1)
y = data["Profit"].values.reshape(m, 1)
theta = np.zeros((2, 1))

print("Initial Cost:", computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.T, (predictions - y))
        theta -= alpha * (1/m) * error
        j_history.append(computeCost(X, y, theta))
    return theta, j_history

theta, j_history = gradientDescent(X, y, theta, 0.01, 1500)

print(f"Model: h(x) = {round(theta[0,0],2)} + {round(theta[1,0],2)}x")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\Theta)$")
plt.title("Cost Function Reduction")
plt.show()

plt.scatter(data["Population"], data["Profit"])
x_line = np.linspace(data["Population"].min(), data["Population"].max(), 100)
y_line = theta[0,0] + theta[1,0] * x_line
plt.plot(x_line, y_line, color="r")
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Linear Regression Fit")
plt.show()

```

## Output:
![linear regression using gradient descent](sam.png)

<img width="774" height="621" alt="Screenshot 2026-02-04 162832" src="https://github.com/user-attachments/assets/09ed1067-5454-4b62-ac21-92e01603a70a" />

<img width="773" height="583" alt="Screenshot 2026-02-04 162854" src="https://github.com/user-attachments/assets/b5157645-d1bf-4be0-97db-f261c2b32248" />

<img width="773" height="606" alt="Screenshot 2026-02-04 162914" src="https://github.com/user-attachments/assets/75cee3e5-687e-4dc9-936f-e74fa6710388" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)

x= [2.632214144151, 3.272355, 2.7607268, 3.4447693, 3.6799583, 3.5055826, 2.9330038, 2.8568107, 3.3276575, 2.8926306, 2.7319663, 3.5326295, 3.160792, 2.8481617, 3.2420505, 3.0637345, 3.0162536, 3.3754508, 3.489457, 2.8902778, 3.2010884]

# generate points used to plot
x_plot = np.array(x)


# generate points and keep a subset of them
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]


colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)



plt.legend(loc='lower left')

plt.show()
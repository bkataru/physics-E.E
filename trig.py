import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score


y = np.array([2.632214144151, 3.272355, 2.7607268, 3.4447693, 3.6799583, 3.5055826, 2.9330038, 2.8568107, 3.3276575, 2.8926306, 2.7319663, 3.5326295, 3.160792, 2.8481617, 3.2420505, 3.0637345, 3.0162536, 3.3754508, 3.489457, 2.8902778, 3.2010884])
x = np.array(list(range(len(y))))

def test_func(x, a, b, c, d):
    return (a * np.sin(b * (x + c))) + d

params, params_covariance = optimize.curve_fit(test_func, x, y,
                                               p0=[2, 2, 2, 2])

print("R^2: {}".format(r2_score(y, test_func(x, params[0], params[1], params[2], params[3]))))

plt.figure(figsize=(6, 4))

plt.scatter(x, y, label='Data point')
plt.plot(x, y, label='Data')
print(params[0], params[1], params[2], params[3])
plt.plot(x, test_func(x, params[0], params[1], params[2], params[3]),
         label='Fitted function')

'''
# prediction section below:
t = np.array(list(range(20, 80)))
plt.plot(t, test_func(t, params[0], params[1], params[2], params[3]),
         label='Predicted trend')
'''

plt.legend(loc='best')
plt.show()



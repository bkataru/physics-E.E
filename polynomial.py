import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y = np.array([2.632214144151, 3.272355, 2.7607268, 3.4447693, 3.6799583, 3.5055826, 2.9330038, 2.8568107, 3.3276575, 2.8926306, 2.7319663, 3.5326295, 3.160792, 2.8481617, 3.2420505, 3.0637345, 3.0162536, 3.3754508, 3.489457, 2.8902778, 3.2010884])
print(len(y))
x = np.array(list(range(len(y))))

n = 17     # the degree of the polynomial

p = np.poly1d(np.polyfit(x, y, n))

t = np.array(list(range(len(y))))
'''
#print(r2_score(y, p(t)))
plt.plot(x, y, 'o', t, p(t), '-')
plt.show()
'''
# prediction section below:

k = np.array(list(range(22, 53)))
plt.plot(x, y, 'o', k, p(k), '-')
plt.show()

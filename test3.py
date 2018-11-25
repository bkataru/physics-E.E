import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = np.array([2.632214144151, 3.272355, 2.7607268, 3.4447693, 3.6799583, 3.5055826, 2.9330038, 2.8568107, 3.3276575, 2.8926306, 2.7319663, 3.5326295, 3.160792, 2.8481617, 3.2420505, 3.0637345, 3.0162536, 3.3754508, 3.489457, 2.8902778, 3.2010884])
t = np.array(list(range(len(data))))

guess_mean = np.mean(data)
guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
guess_phase = 0
guess_freq = 1
guess_amp = 1

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]


# recreate the fitted curve using the optimized parameters

fine_t = np.arange(0,max(t),0.1)
data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean

print(r2_score(data, data_fit[:21]))

plt.plot(t, data, '.')
plt.plot(t, data_first_guess, label='first guess')
plt.plot(fine_t, data_fit, label='after fitting')
plt.legend()
plt.show()
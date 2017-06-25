from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 1, 500, endpoint=False)

plt.figure()
sig = np.sin(2 * np.pi * t)
pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.subplot(2, 1, 2)
plt.plot(t, pwm)
plt.ylim(-1.5, 1.5)

plt.show()
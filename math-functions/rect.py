#as well as pwm.py from:
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.square.html

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 1, 500, endpoint=False)
plt.plot(t, signal.square(2 * np.pi * 5 * t))
plt.ylim(-2, 2)
plt.show()

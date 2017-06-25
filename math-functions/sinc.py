import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np


#Fs = 100
#f = 5
#sample = 10
#x = np.arange(sample)
#y = np.sin(2 * np.pi * f * x / Fs)
#plt.plot(x, y)
#plt.xlabel('voltage(V)')
#plt.ylabel('sample(n)')

T = 1
t = np.arange(-7.0, 7.0, 0.01)
s = T*np.sin(2 *  t * T)/(np.pi*t*T)
fig, ax = plt.subplots()
ax.plot(t, s)
ax.grid(True)

#source: http://matplotlib.org/examples/pylab_examples/axes_props.html
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

plt.show()
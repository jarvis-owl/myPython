
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

print("imported np, tf, and matplot")

for i in range(100):
	if not i%10:
		print(i)
x = np.arange(10)
y = x**2
plt.plot(x,y,linewidth=2.0)
plt.show()
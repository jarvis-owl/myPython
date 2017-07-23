import numpy as np
import matplotlib.pyplot as plt

#scores = [3.0, 1.0, 0.2] # not working with np.shape()
scores = [[3.0, 1.0, 0.2],[4.0, 0.3, 1.6]]

"""
def mySoftmax(x):
	columns, rows = np.shape(x)
	#print(columns,rows,x[])
	sum=0
	for j in range(columns):
		for i in range(rows):
        	sum += x[i][j]
	sum := rows*columns
	pass sum
"""
#SOLUTION
"""Softmax."""

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x),axis=0)


print(softmax(scores*10))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores/10).T, linewidth=2)
plt.show()

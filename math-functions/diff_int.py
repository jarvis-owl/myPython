import matplotlib as plz
#import math
from sympy import *
#import numpy as np

x = Symbol('x')
y = Symbol('y')

print('derivative: ')
print(diff(sin(x),x))

print('integral')
print(integrate(sin(x), (x, 0, pi/2)))

#source: https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
#other integrations are available such as trapz for samples

from scipy.integrate import quad
def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print('integral area: ',I[0])
print('abs error: ',I[1])

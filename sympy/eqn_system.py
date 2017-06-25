from sympy import *

a = Symbol('a')
T = Symbol('T')

f1 = Eq((1-a)/(2*T),12.4e3)
f2 = Eq((1+a)/(2*T),31.6e3)

#print(solve(f1,a))
#print(solve(f2,a))


ans = solve([f1,f2],(a,T))

#print(ans)
#print(ans['a'])
#print(ans.items())
#print(ans.keys())
#print(ans.values())
#print()
a = ans[a]
print("a = ",a)
T = ans[T]
print("T = %10.3e" % T ,"s")


"""
    24.07.'17
    jarvis
    v1

    calculating cablelength vs 12V transformation

"""
import numpy as np

A = 0.6     #[mm²]
r = 0.017   #[Ohm mm² / m]
l = 5       #[m]
P_a = 5     #[W]
n_1 = 0.9   #efficiency
n_2 = 0.9
U_h = 12    #[V]
U_l = 5     #[V]


I = 1       #[A](U)

#voltage drop
U_d= 2 * l * I/(1/r*A)
print(U_d)

R = r * l / A
print(R)

P_vk = U_d**2 /R #Wrong - delta U is right, but unknown
print(P_vk)

P_e = (P_a/n_2 - P_vk)*1/n_1
print(P_e)

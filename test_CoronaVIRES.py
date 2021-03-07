"""
Play with the model
"""
from CoronaVIRES_1 import CoronaVIRES_1

print("Testing CoronaVIRES_1")
US_pop = 382_000_000
N = US_pop
model = CoronaVIRES_1(N)
# model.predict_Deaths(10,alpha=1.2*10**(-4),beta = 0.1, del1 = 0.01, del2=0.001, chi=0.16, dels = 0.03, rho = 1.33/380, phi = 1/30, phi2 = 1/15, theta = 0.001, Es0 = )
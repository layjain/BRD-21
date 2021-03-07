from CoronaVIRES_1 import CoronaVIRES_1
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import numpy as np

np.seterr("raise")

owid_df = pd.read_csv("owid/owid-covid-data-new.csv")

country_location = "Italy"
owid_country = owid_df.loc[owid_df['location']==country_location]
# Filter relavant dates since vaccinations started (is Not NAN)
owid_country = owid_country[owid_country.total_vaccinations.notnull()]

#Series to Predict
def date_difference(s1,s2):
    #s1-s2
    y1,m1,d1 = s1.split("-")
    y1,m1,d1 = int(y1), int(m1), int(d1)

    y2,m2,d2 = s2.split("-")
    y2, m2, d2 = int(y2), int(m2), int(d2)

    f_date = date(y1, m1, d1)
    l_date = date(y2, m2, d2)

    return (f_date-l_date).days

_deaths = list(owid_country.total_deaths)
deaths = [e-_deaths[0] for e in _deaths]

_dates = list(owid_country.date)
dates = [date_difference(e, _dates[0]) for e in _dates]
populations = {"United States":328_000_000, "Italy":60_360_000}
N = populations[country_location]

# Models
model_1 = CoronaVIRES_1(N)
def f1(t,alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
    ret = []
    for T in t:
        death_T = model_1.predict_Deaths(int(T), alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
        ret.append(death_T)
    return ret
# alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
lower_bounds = [0,0,0,0,0,0,0,0,0,0,N//2,N//10000,N//10000]
#               alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
upper_bounds = [1,     0.5,   0.3,  0.2,   1,   0.1, 0.1, 0.2, 1,    0.5,   N,   N,  N//10]
opt = curve_fit(f1, dates, deaths, bounds = (lower_bounds,upper_bounds))

#Plot
alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0 = opt[0]
model_final_1 = CoronaVIRES_1(N)
T = max(dates)
model_final_1.run_predict(T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
plt.scatter(dates, [model_final_1.D[i] for i in dates], label = "Predicted 1")
plt.scatter(dates, deaths, label="Deaths_Actual")
plt.legend()
plt.show()
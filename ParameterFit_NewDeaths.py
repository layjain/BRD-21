from CoronaVIRES_1 import CoronaVIRES_1
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from utils import *

np.seterr("raise")

owid_df = pd.read_csv("owid/owid-covid-data-new.csv")

"""
Get the top few countries
"""
country_to_vaccination_days_count = {}
for country_location in owid_df["location"].unique(): 
#     country_location = "United States"
    owid_country=owid_df.loc[owid_df['location']==country_location]
    count = owid_country["total_vaccinations"].dropna().count()
    country_to_vaccination_days_count[country_location] = count

# See top 10 countries data we have data for
top_few = dict(sorted(country_to_vaccination_days_count.items(), key=itemgetter(1), reverse=True)[:10])

for country_location in list(top_few)+["United States"]:
    # Example: country_location = "Italy"
    owid_country = owid_df.loc[owid_df['location']==country_location]

    # Filter relavant dates since vaccinations started (is Not NAN)
    owid_country = owid_country[owid_country.total_vaccinations.notnull()]

    #Series to Predict
    new_deaths = list(owid_country.new_deaths_smoothed)
    print(country_location)

    _dates = list(owid_country.date)
    dates = [date_difference(e, _dates[0]) for e in _dates]

    N = list(owid_country["population"])[0]

    # Models
    model_1 = CoronaVIRES_1(N)
    def f1(t,alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
        ret = []
        for T in t:
            death_T = model_1.predict_new_deaths(int(T), alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
            ret.append(death_T)
        return ret
    
    # alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
    lower_bounds = [0,0,0,0,0,0,0,0,0,0,N//2,N//10000,N//10000]
    #               alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
    upper_bounds = [1,     0.5,   0.3,  0.2,   1,   0.1, 0.1, 0.2, 1,    0.5,   N,   N,  N//10]
    opt = curve_fit(f1, dates, new_deaths, bounds = (lower_bounds,upper_bounds))

    #Plot
    alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0 = opt[0]
    model_final_1 = CoronaVIRES_1(N)
    T = max(dates)
    model_final_1.predict_new_deaths(T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
    plt.scatter(dates, [model_final_1.new_deaths[i] for i in dates], label = "Predicted 1", marker='.')
    plt.scatter(dates, new_deaths, label="New Deaths Actual", marker='.')
    # plt.scatter(owid_country.date, owid_country.total_vaccinations, label="Total Vaccinations")
    plt.title(country_location)
    plt.legend()
    plt.show()
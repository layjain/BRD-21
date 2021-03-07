from CoronaVIRES_1 import CoronaVIRES_1
from SEIR_1 import SEIR_Baseline
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from utils import *

TAU = 0.9

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

results_df = pd.DataFrame()
results_df["Parameter/Countries"] = ["alpha", "beta", "del1", "del2", "chi", "dels", "rho", "phi", "phi2", "theta", "S0", "Es0", "Is0"]
for country_location in list(top_few)+["United States"]:
    # Example: country_location = "Italy"
    owid_country = owid_df.loc[owid_df['location']==country_location]

    # Filter relavant dates since vaccinations started (is Not NAN)
    owid_country = owid_country[owid_country.total_vaccinations.notnull()]

    #Series to Predict
    N = list(owid_country["population"])[0]

    _deaths = list(owid_country.total_deaths)
    deaths = [e-_deaths[0] for e in _deaths]
    deaths = [death*1e6/N for death in deaths]  # standardize the deaths
    train_deaths = deaths[:int(TAU*len(deaths))]

    _dates = list(owid_country.date)
    dates = [date_difference(e, _dates[0]) for e in _dates]
    train_dates = dates[:int(TAU*len(deaths))]
    
    N = 1e6
    # Models
    model_1 = CoronaVIRES_1(N)
    model_base = SEIR_Baseline(N)

    # def f1(t,alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
    #     ret = []
    #     for T in t:
    #         death_T = model_1.predict_Deaths(int(T), alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
    #         ret.append(death_T)
    #     return ret
    
    def f2(t,alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
        predicted_deaths = model_1.predict_Deaths_for_T_days(int(max(t)), alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
        ret = []
        for time in t:
            ret.append(predicted_deaths[int(time)])
        return ret

    # alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
    lower_bounds = [0,0,0,0,0,0,0,0,0,0,N//2,N//10000,N//10000]
    # lower_bounds = [0,0,0,0,0,0,0,0,0,0,N//3,N//100000,N//100000]
    #               alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0
    # upper_bounds = [1,     0.5,   0.3,  0.2,   1,   0.1, 0.1, 0.2, 1,    0.5,   N,   N,  N//10]
    upper_bounds = [1, 0.5, 0.3, 0.2, 1, 0.1, 0.1, 0.2, 1, 0.5, N, N, N]
    opt = curve_fit(f2, dates, deaths, bounds = (lower_bounds,upper_bounds))
    

    def f1_base(t,alpha, beta, chi, dels, rho, theta, S0, Es0, Is0):
        ret = []
        for T in t:
            death_T = model_base.predict_Deaths(int(T), alpha, beta, chi, dels, rho, theta, S0, Es0, Is0)
            ret.append(death_T)
        return ret

    lower_bounds_base = [0,        0,  0,   0,    0,     0,  N//3, N//100000,N//100000]
    #               alpha, beta, chi, dels, rho, theta, S0, Es0, Is0
    # upper_bounds = [1,     0.5,   0.3,  0.2,   1,   0.1, 0.1, 0.2, 1,    0.5,   N,   N,  N//10]
    upper_bounds_base = [1, 0.5, 1, 0.1, 0.1,      0.5, N, N, N]
    opt_base = curve_fit(f1_base, dates, deaths, bounds = (lower_bounds_base,upper_bounds_base))



    #Plot
    alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0 = opt[0]
    alpha_base, beta_base, chi_base, dels_base, rho_base, theta_base, S0_base, Es0_base, Is0_base = opt_base[0]

    model_final_1 = CoronaVIRES_1(N)
    T = max(dates)
    model_final_1.run_predict(T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)

    model_base_final = SEIR_Baseline(N)
    model_base_final.run_predict(T, alpha_base, beta_base, chi_base, dels_base, rho_base, theta_base, S0_base, Es0_base, Is0_base)
    print("Fitting Done,Plotting")
    plt.scatter(dates, [model_final_1.D[i] for i in dates], label = "Predicted 1", marker='.')
    plt.scatter(dates, [model_base_final.D[i] for i in dates], label = "SEIR Baseline", marker='.')
    plt.scatter(dates, deaths, label="Deaths Actual", marker='.')
    plt.xticks([train_dates[-1]], ['End of Training data']) 
    plt.axvline(x=train_dates[-1], ymin=0, ymax=1, linestyle = "dashed")
    # plt.scatter(owid_country.date, owid_country.total_vaccinations, label="Total Vaccinations")
    plt.title(country_location)
    plt.legend()
    plt.show()
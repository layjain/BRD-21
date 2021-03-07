from datetime import date

def date_difference(s1,s2):
    #s1-s2
    y1,m1,d1 = s1.split("-")
    y1,m1,d1 = int(y1), int(m1), int(d1)

    y2,m2,d2 = s2.split("-")
    y2, m2, d2 = int(y2), int(m2), int(d2)

    f_date = date(y1, m1, d1)
    l_date = date(y2, m2, d2)

    return (f_date-l_date).days

def calculate_errors(deaths,train_deaths,predicted_deaths,predicted_deaths_base):
    train_len = len(train_deaths)
    total_len = len(deaths)
    sse_baseline = 0 #Sum-Squared-Errors
    sse_coronavires = 0
    for i in range(train_len,total_len):
        baseline_pred = predicted_deaths_base[i]
        coronavires_predict = predicted_deaths[i]
        actual_death = deaths[i]
        sse_baseline += (baseline_pred - actual_death)**2/(total_len-train_len)
        sse_coronavires += (coronavires_predict - actual_death)**2/(total_len - train_len)
    return (sse_coronavires**0.5, sse_baseline**0.5)

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
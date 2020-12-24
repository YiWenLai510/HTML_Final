import pandas as pd

if __name__ == "__main__":
    feature = pd.read_csv("train.csv").rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    feature["month"].replace({'January':1,
                                           'February':2,
                                           'March':3,
                                           'April':4,
                                           'May':5,
                                           'June':6,
                                           'July':7,
                                           'August':8,
                                           'September':9,
                                           'October':10,
                                           'November':11,
                                           'December':12}, inplace=True)
    feature["date"] = pd.to_datetime(feature[["year","month","day"]])
    feature = feature.drop(columns=["year","month","day"]).sort_values("date")
    
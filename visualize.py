import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    feature = pd.read_csv("train.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    testFeature = pd.read_csv("test.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    testCanceled = pd.read_csv("cancel.csv")
    # print(testCanceled)
    testFeature["is_canceled"] = testCanceled["label"]
    # print(testFeature)
    feature = pd.concat([feature,testFeature], ignore_index=True)
    # print(feature)
    feature = feature[feature["is_canceled"] == 0]
    # feature["hotel"].replace({"Resort Hotel":1,"City Hotel":0}, inplace=True)
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
    # feature["meal"].replace({"Undefined":1,"SC":1,"BB":2,"HB":3,"FB":4}, inplace=True)
    # feature["deposit_type"].replace({"No Deposit":1,"Non Refund":2,"Refundable":3}, inplace=True)
    feature["desired_room"] = (feature["reserved_room_type"] == feature["assigned_room_type"])
    feature["days"] = feature["stays_in_weekend_nights"] + feature["stays_in_week_nights"]
    feature["date"] = pd.to_datetime(feature[["year","month","day"]])
    feature.drop(columns=["year","month","day","ID","is_canceled","reservation_status","reservation_status_date",
                          "reserved_room_type","stays_in_weekend_nights","stays_in_week_nights"], axis=1, inplace=True)
    # print(len(feature["market_segment"]))
    toEncode = ["country","market_segment","distribution_channel","assigned_room_type","agent","company","customer_type"] + ["hotel","meal","deposit_type"]
    for f in toEncode:
        encoder = LabelBinarizer()
        binaryFeature = pd.DataFrame(encoder.fit_transform(feature[f].astype(str)))
        # sleep(10)
        feature = pd.concat([feature.reset_index(drop=True), binaryFeature.reset_index(drop=True)], axis=1).drop([f], axis=1)
    # print(feature)

    train = feature[feature["date"] < pd.to_datetime("2017-04-1")]
    test = feature[feature["date"] > pd.to_datetime("2017-03-31")]
    
    adrLabel = train["adr"]
    # print(adrLabel)
    adrFeature = train.drop(columns=["adr","days","date"], axis=1)
    # print(adrFeature)
    
    pca = PCA(n_components=2).fit(adrFeature) 
    adrFeature = pca.transform(adrFeature)

    trans = pd.DataFrame()
    trans['x'] = adrFeature[:,0]
    trans['y'] = adrFeature[:,1]
    trans["adr"] = adrLabel

    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="x", y="y",
    hue="adr",
    palette=sns.color_palette("hls",as_cmap=True),
    data=trans,
    legend="full",
    alpha=0.3
    )

    # plt.show()
    plt.savefig('hotel2D.png')
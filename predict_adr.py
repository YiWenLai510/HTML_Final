import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def lightgbm(y, x, test_x):
    # preprocess data
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    test_x = np.array(test_x).astype(float)
    index_list = np.arange(x.shape[0])
    np.random.shuffle(index_list)
    unit_len = int(x.shape[0]/5)
    total_data = lgb.Dataset(x, label=y)
    train_data = lgb.Dataset(x[index_list[unit_len:]], label=y[index_list[unit_len:]])
    validation_data = lgb.Dataset(x[index_list[:unit_len]], label=y[index_list[:unit_len]])

    print('test')
    print(x.shape)
    print(y.shape)
    params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'n_estimators': 120,
    'learning_rate': 0.1, 
    'num_leaves': 31, 
    'max_depth': 6,
    'reg_lambda': 0,
    }
    params['metric'] = 'rmse'
    bst = lgb.train(params, train_data, valid_sets=[validation_data], early_stopping_rounds=5)
    y_pred = bst.predict(test_x, num_iteration=bst.best_iteration)

    return y_pred

def predict_adr(df, n_neighbors):  # add neighbor param

    drop = ["stays_in_weekend_nights", "stays_in_week_nights","date", "reserved_room_type"]
    toEncode = ["country", "market_segment", "distribution_channel", "assigned_room_type", "agent", "company",
                "customer_type","hotel", "meal", "deposit_type"]
    df.read_drop_encode(drop, toEncode, [])
    df.drop_encode()

    adrLabel, adrFeature, testf = df.getTrainTest_adr_pd()
    print(adrLabel.shape)
    print(adrFeature.shape)
    print(testf.shape)
    # print(adrFeature)

    # pca = PCA(n_components=85).fit(adrFeature)
    # adrFeature = pca.transform(adrFeature)

    # model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(adrFeature, adrLabel)
    # 1.539474

    # model = LinearRegression().fit(adrFeature, adrLabel)
    # 0.618421

    # model = SVR().fit(adrFeature, adrLabel)
    # 0.776316

    # model = MLPRegressor(max_iter=500).fit(adrFeature, adrLabel)
    # 0.486842

    # print(model.score(adrFeature, adrLabel))
    # _predict = model.predict(adrFeature)

    # print((_predict-adrLabel).abs().mean())
    # dump(model, 'sklearn_adr_KNR')
    # model = load('sklearn_adr_KNR')
    # print('score', model.score(adrFeature, adrLabel))
    # testf = test.drop(columns=["adr", "days", "date"], axis=1)
    #test_pca = pca.transform(testf)

    model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(adrFeature, adrLabel)
    # 0.486842
    dump(model, 'sklearn_adr_KNR')
    print('score', model.score(adrFeature, adrLabel))
    return model.predict(testf)

    # y_pred = lightgbm(adrLabel, adrFeature, testf)
    # return y_pred


import numpy as np
import pandas as pd
class DataReader(object):
    def __init__(self, trainFile, testFile, notinclude, drop_canceled, notOneHot,toOneHot):
        self.trainRow = pd.read_csv(trainFile).fillna(value=0).rename(columns={"arrival_date_year": "year", "arrival_date_month": "month", "arrival_date_day_of_month": "day"})
        
        # drop the negative adr
        self.trainRow = self.trainRow[self.trainRow['adr'] > 0]
        # drop stay 0 nights
        self.trainRow = self.trainRow[(self.trainRow['stays_in_weekend_nights'] != 0) | (self.trainRow['stays_in_week_nights'] != 0)]
        # drop 0 people
        self.trainRow = self.trainRow[(self.trainRow['adults'] != 0) | (self.trainRow['children'] != 0) | (self.trainRow['babies'] != 0)]
        
        print(self.trainRow)
        print(self.trainRow[(self.trainRow['adults'] == 0) & (self.trainRow['children'] == 0) & (self.trainRow['babies'] == 0)])
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('trainRow before', self.trainRow)
        # print('trainRow after', self.trainRow_adr)    
        
        self.train_cancel = self.trainRow['is_canceled']
        self.train_adr = self.trainRow['adr']
        self.testRow = pd.read_csv(testFile).fillna(value=0).rename(columns={"arrival_date_year": "year", "arrival_date_month": "month", "arrival_date_day_of_month": "day"})

        self.not_included = notinclude
        self.drop = drop_canceled
        self.not_onehot = notOneHot
        self.encode = toOneHot

        self.trainStartIndex = 0
        self.testStartIndex = len(self.trainRow.index)  # check
        self.trainEndIndex = (self.testStartIndex) - 1

        self.wholeData = self.combined()
        print("====== First initialize Whole Data ===============")
        print (self.wholeData)
    def read_drop_encode(self,drop,encode,notencode):
        self.drop = drop
        self.encode = encode
        self.not_onehot = notencode
    def drop_encode(self):
        # remove unwanted columns, drop_canceled
        # cancel: ['agent', 'country', 'arrival_date_week_number',"year", "month", "day","date"]
        tmp_combined = self.wholeData.drop(columns=self.drop)
        # print('tmp_combined =======================')
        # print(tmp_combined)
        # print("column",list(tmp_combined.columns.values))
        # columns be encode
        # cancel:notoneHot = ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','days','total_of_special_requests']
        encode_col = []
        if len(self.encode) == 0:
            # print("here====")
            all_columns = list(tmp_combined.columns.values)
            for c in all_columns:
                if c not in self.not_onehot:
                    encode_col.append(c)
        else:
            print("second time =====================")
            print(tmp_combined)
            encode_col = self.encode
        # print("encode",encode_col)
        # one hot encoder
        self.readytoTrain = pd.get_dummies(tmp_combined, columns=encode_col)
        print(self.readytoTrain)
    def combined(self):
        # remove tags in train but not in test
        # not_include = ['ID', 'is_canceled','adr' ,'reservation_status' ,'reservation_status_date']
        tmp_train = self.trainRow.drop(columns=self.not_included)
        tmp_test = self.testRow.drop(columns=['ID'])
        # print(tmp_train)
        # print(tmp_test)
        # combine train and test
        tmp_combined = pd.concat([tmp_train, tmp_test])
        # # z-score normalization (lead_time)
        # mu = tmp_combined['lead_time'].mean()
        # std = tmp_combined['lead_time'].std()
        # tmp_combined['lead_time'] = (tmp_combined['lead_time'] - mu) / std

        tmp_combined['month'].replace({   'January': 1,
                                          'February': 2,
                                          'March': 3,
                                          'April': 4,
                                          'May': 5,
                                          'June': 6,
                                          'July': 7,
                                          'August': 8,
                                          'September': 9,
                                          'October': 10,
                                          'November': 11,
                                          'December': 12},inplace=True)
        tmp_combined["desired_room"] = (tmp_combined["reserved_room_type"] == tmp_combined["assigned_room_type"])
        tmp_combined["days"] = tmp_combined["stays_in_weekend_nights"] + tmp_combined["stays_in_week_nights"]
        tmp_combined["date"] = pd.to_datetime(tmp_combined[["year", "month", "day"]])
        tmp_combined = tmp_combined.drop(columns=["year", "month", "day"])
        return tmp_combined


    def add_column_to_test(self, predict_result):
        if len(self.testRow.index) == len(predict_result):
            self.testRow['is_canceled'] = predict_result
            # print(self.testRow)
            self.wholeData['is_canceled'] = np.concatenate((predict_result,self.train_cancel.to_numpy()))
            print ("add cancel ================================================")
            print(self.wholeData)
        else:
            print(len(predict_result))
            print('Error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def getTrainTest_cancel_np(self):
        df_to_numpy = self.readytoTrain.to_numpy()
        return np.array(self.train_cancel).astype(int), (df_to_numpy[:self.trainEndIndex + 1]).astype(float), (
        df_to_numpy[self.testStartIndex:]).astype(float)
        #     train = feature[feature["date"] < pd.to_datetime("2017-04-1")]
        #     test = feature[feature["date"] > pd.to_datetime("2017-03-31")]
    def getTrainTest_adr_pd(self):
        df_to_numpy = self.readytoTrain.to_numpy()
        # print(self.readytoTrain)
        return self.train_adr,self.readytoTrain[:self.trainEndIndex + 1],self.readytoTrain[self.testStartIndex:]
        #return self.train_adr.to_numpy(), (df_to_numpy[:self.trainEndIndex + 1]), (df_to_numpy[self.testStartIndex:])
import pandas as pd
import numpy as np
from libsvm.svmutil import *
class DataReader(object):    
    def __init__(self,trainFile,testFile,notinclude,drop_canceled,notoneHot):
        self.trainRow = pd.read_csv(trainFile)
        self.train_label = self.trainRow['is_canceled']
        self.testRow = pd.read_csv(testFile)

        self.not_included = self.ignored_parser(notinclude)
        self.drop_canceled = drop_canceled
        self.not_onehot = notoneHot

        self.wholeData = self.combined()
        print(self.wholeData)
        self.trainStartIndex = 0
        self.testStartIndex = len(self.trainRow.index) # check
        self.trainEndIndex = (self.testStartIndex) - 1
        
    def combined(self):
        # remove tags in train but not in test
        tmp_train = self.trainRow.drop(columns = self.not_included) # contain ID
        tmp_test = self.testRow.drop(columns = ['ID'])
        # combine train and test
        tmp_combined = pd.concat([tmp_train,tmp_test])
        # remove unwanted columns
        tmp_combined = tmp_combined.drop(columns=self.drop_canceled)
        # columns be encode
        encode_col = []
        all_columns = list(tmp_combined.columns.values)
        for c in all_columns:
            if c not in self.not_onehot:
                encode_col.append(c)
        # one hot encoder
        return pd.get_dummies(tmp_combined.astype(str),columns=encode_col)
    # def add_column_to_test(self):
    def ignored_parser(self,ignore):
        return ignore.split()
    def getTrain_svm(self):
        df_to_list = self.wholeData.values.tolist()
        # print(len(df_to_list))
        # print(self.train_label[0])
        # print(self.wholeData.iloc[0])
        # print(len(df_to_list[0]))
        with open('train_svm.txt', 'w') as f:
            for i in range(len(df_to_list)):
                if i <= self.trainEndIndex:
                    row_str = ''
                    row_str += str(self.train_label[i])
                    for index,value in enumerate(df_to_list[i]):
                        row_str += ' %d:%s'%(index,value)
                    row_str += '\n'
                    print(i)
                    f.write(row_str)
                else:
                    break

        # find how to return specific range of data and change to csv


    def getTest_svm(self):
        df_to_list = self.wholeData.values.tolist()
        with open('test_svm.txt', 'w') as f:
            for i in range(len(df_to_list)):
                if i >= self.testStartIndex:
                    row_str = ''
                    row_str += '0'
                    for index,value in enumerate(df_to_list[i]):
                        row_str += ' %d:%s'%(index,value)
                    row_str += '\n'
                    print(i)
                    f.write(row_str)


def processData(train,test,notinclude,drop_canceled,notoneHot):
    df = DataReader(train,test,notinclude,drop_canceled,notoneHot)
    df.getTrain_svm()
    # df.getTest_svm()
    # return df,train_svm_format,test_svm_format

def train_svm(train):
    y, x = svm_read_problem(train)
    model = svm_train(y, x, '-c 10 -s 0 -t 0')

    print('complete train')
    p_label, p_acc, p_val = svm_predict(y, x, model)
    return model

# def predict_and_add_iscanceled(model,test,df_obj):
#     y, x = svm_read_problem(test)
#     p_label, p_acc, p_val = svm_predict([], x, model) # check problem  first svm predict with no test label
#     testDataframe = df_obj.add_column_to_test(p_label,'is_canceled')
#     return testDataframe
    
if __name__ == '__main__':
    # make train & test same dimension
    notinclude = 'ID is_canceled adr reservation_status reservation_status_date'
    # should be preserved not be encode
    notoneHot =  ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies']
    # redunancy info for predict canceled
    drop_canceled = ['agent', 'arrival_date_month', 'country', 'arrival_date_week_number', 'arrival_date_day_of_month']

    processData('train.csv','test.csv',notinclude,drop_canceled,notoneHot)
    # m = train_svm('train_svm.txt')
    # predict_and_add_iscanceled(m, testing, df_Obj)
    # print(label)
    # model = train(dataframe,label)
    # predict(model,label)


    # def processData(self):
    #     self.train_dataframe = self.data.copy()
    #     self.all_need_attributes = []
    #     all_attributes = self.train_dataframe.columns.values
    #     for tag in all_attributes:
    #         if tag not in self.not_included:
    #             self.all_need_attributes.append(tag)
    #             if self.train_dataframe[tag].dtype == 'object':
    #                 self.build_dict(tag)
    #     self.train_dataframe = self.train_dataframe.drop(columns = self.not_included)
    # def build_dict(self,tag):
    #     a = []
    #     d = {}
    #     col = self.train_dataframe[tag]
    #     for x in col:
    #         if x not in a:
    #             a.append(x)
    #     for index,value in enumerate(a):
    #         d[value] = index
    #     self.train_dataframe[tag] = self.train_dataframe[tag].replace(d)
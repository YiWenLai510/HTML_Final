import pandas as pd
import numpy as np
from svmutil import *
    
class Trainer(object):
    def __init__(self, trainFile, trainLabel, testFile, testLabel, notinclude, drop_canceled, notoneHot):
        self.trainRow = pd.read_csv(trainFile)
        self.labelRow = pd.read_csv(trainLabel)

        # classify data with date
        self.order_with_date = self.classfiler()
        print('self.order_with_date:', len(self.order_with_date))

    # combine the data in the same date and isn't canceled
    def classfiler(self):
        dateList = self.labelRow['arrival_date'].tolist()
        print('arrival_date:', len(dateList))
        order_with_date = []
        for i in range(len(dateList)):
            date = dateList[i].split('-')
            year = date[0]
            month = date[1]
            day = date[2]
            trainRow_1_day = self.trainRow.loc[(self.trainRow['arrival_date_year'] == int(year)) & \
                (self.trainRow['arrival_date_month'] == self.month_to_number(int(month))) & \
                (self.trainRow['arrival_date_day_of_month'] == int(day)) & \
                (self.trainRow['is_canceled'] != 1)]
            order_with_date.append(trainRow_1_day)
        return order_with_date
        
    def month_to_number(self, month):
        return {
            1 : 'January',
            2 : 'Febuary',
            3 : 'March',
            4 : 'April',
            5 : 'May',
            6 : 'June',
            7 : 'July',
            8 : 'August',
            9 : 'September', 
            10 : 'October',
            11 : 'November',
            12 : 'December'
        }[month]

    # compute the total adr amoung the orders in one day
    def get_adr(self, trainRow_1_day):
        adr = trainRow_1_day['adr'].tolist()
        total_adr = 0
        for i in range(len(adr)):
            total_adr += adr[i]
        return total_adr

    # predict with total adr and check with label
    def predict_with_adr(self, label = None):
        adr = []
        for i in range(len(self.order_with_date)):
            adr.append(self.get_adr(self.order_with_date[i]))
        rank = self.rank(adr, 'linear')
        print('adr:', len(adr))
        print('rank:', len(rank))

        if label is not None:
            err = self.loss_function(rank, label)
            err = str(err) + '/' + str(len(rank))
            return rank, err

        return rank

    # predict rank with adr  (can add many method or use neural network?)
    def rank(self, adr, method = 'linear'):
        if method == 'linear':
            min_adr = min(adr)
            max_adr = max(adr)
            class_interval_width = (max_adr - min_adr) / 10
            class_interval = [min_adr + class_interval_width * i for i in range(10)]
            rank = []
            for i in range(len(adr)):
                for j in range(len(class_interval)-1, -1, -1):
                    if adr[i] >= class_interval[j]:
                        rank.append(j)
                        break
            return rank
        else:
            pass
        
    # compute loss
    def loss_function(self, rank, label, method = 'zero_one'):
        if method == 'zero_one':
            print(len(rank))
            print(len(label))
            assert len(rank) == len(label)
            err = 0
            for i in range(len(rank)):
                if int(rank[i]) != int(label[i]):
                    err += 1
            return err
        else:
            pass
            


if __name__ == '__main__':
    # make train & test same dimension
    notinclude = ['ID', 'is_canceled' ,'adr' ,'reservation_status' ,'reservation_status_date']
    # should be preserved not be encode
    notoneHot =  ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies']
    # redunancy info for predict canceled
    drop_canceled = ['agent', 'arrival_date_month', 'country', 'arrival_date_week_number', 'arrival_date_day_of_month']# 'agent',


    # find the relationship between adr and label
    test = Trainer('train.csv', 'train_label.csv', 'test.csv', 'test_nolabel.csv', notinclude, drop_canceled, notoneHot)
    rank, e_in = test.predict_with_adr(test.labelRow['label'].tolist())
    print('rank predict with adr:', rank)
    print('Ein with linear:', e_in)







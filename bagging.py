import pandas as pd

beg = []
# adaboost 120
beg.append(pd.read_csv('result_bagging_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_16.csv')['label'].tolist())

# adaboost 125
beg.append(pd.read_csv('result_bagging_7.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_9.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_11.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_13.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_15.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_17.csv')['label'].tolist())

# gradientboost 120
beg.append(pd.read_csv('result_bagging_grad_7.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_9.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_11.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_13.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_15.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_17.csv')['label'].tolist())

# gradientboost 115
beg.append(pd.read_csv('result_bagging_grad_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_16.csv')['label'].tolist())

# gradientboost 125
beg.append(pd.read_csv('result_bagging_grad_5.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_18.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_20.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_30.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_grad_50.csv')['label'].tolist())

# adaboost 120 add country、week_number
beg.append(pd.read_csv('result_bagging_drop_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_16.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_20.csv')['label'].tolist())

# adaboost 120 add agent week/weekend night
beg.append(pd.read_csv('result_bagging_drop_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_16.csv')['label'].tolist())
beg.append(pd.read_csv('result_bagging_drop_20.csv')['label'].tolist())

label = []
for i in range(len(beg[0])):
    count = [0]*10
    for j in range(len(beg)):
        count[beg[j][i]] += 1
    label.append(count.index(max(count)))


blank_label = pd.read_csv("test_nolabel.csv")
blank_label["label"] = label
blank_label.to_csv('bagging_result.csv',index=False)    
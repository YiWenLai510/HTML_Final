import pandas as pd

beg = []
# adaboost 120
beg.append(pd.read_csv('result_begging_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_16.csv')['label'].tolist())

# adaboost 125
beg.append(pd.read_csv('result_begging_7.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_9.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_11.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_13.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_15.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_17.csv')['label'].tolist())

# gradientboost 120
beg.append(pd.read_csv('result_begging_grad_7.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_9.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_11.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_13.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_15.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_17.csv')['label'].tolist())

# gradientboost 115
beg.append(pd.read_csv('result_begging_grad_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_16.csv')['label'].tolist())

# gradientboost 125
beg.append(pd.read_csv('result_begging_grad_5.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_18.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_20.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_30.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_grad_50.csv')['label'].tolist())

# adaboost 120 add country、week_number
beg.append(pd.read_csv('result_begging_drop_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_16.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_20.csv')['label'].tolist())

# adaboost 120 add agent week/weekend night
beg.append(pd.read_csv('result_begging_drop_6.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_8.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_10.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_12.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_14.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_16.csv')['label'].tolist())
beg.append(pd.read_csv('result_begging_drop_20.csv')['label'].tolist())

label = []
for i in range(len(beg[0])):
    count = [0]*10
    for j in range(len(beg)):
        count[beg[j][i]] += 1
    label.append(count.index(max(count)))


blank_label = pd.read_csv("test_nolabel.csv")
blank_label["label"] = label
blank_label.to_csv('begging_result.csv',index=False)    
import pandas as pd

result_best = pd.read_csv('result_best.csv')
result = pd.read_csv('begging_result.csv')

label = result['label'].tolist()
label_best = result_best['label'].tolist()

count = []
print('result_best   result')
for i in range(len(label)):
    if label[i] != label_best[i]:
        print(i, ':', label_best[i], '-->', label[i])
        count.append(i)

print(len(count))
print(count)

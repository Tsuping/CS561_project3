

with open('test_predictions.csv', 'r') as t1:
    pred_csv = t1.readlines()
with open('test_label.csv', 'r') as t2:
    answer_csv = t2.readlines()

pointer = 0
count = 0
while pointer < len(pred_csv):
    if pred_csv[pointer] == answer_csv[pointer]:
        count += 1
    else:
        print("different: ", answer_csv[pointer] , " your pred: ", pred_csv[pointer])
    pointer += 1
print(count/len(answer_csv))
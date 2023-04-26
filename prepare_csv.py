# -*- coding: utf-8 -*-
import os
import csv
import random
import numpy as np

X = []
age = []

# When the dataset file name comes with its own label
'''
data_root = os.listdir('./datasets/FGNET/')  # Extract images from csv files and make folders as data_root

for image_name in data_root:
    label = os.path.splitext(image_name)[0][-2:]
    if label.isdigit():
        temp = [image_name, label]
        temp2 = [label]
        X.append(temp)
        age.append(temp2)
    else:
        label = os.path.splitext(image_name)[0][-3:-1]
        temp = [image_name, label]
        temp2 = [label]
        X.append(temp)
        age.append(temp2)
'''
# print(len(age))
# print(age)

# The situation where csv and pictures are separated
with open('./datasets/ChaLearn15/Train.csv', 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        temp = [i[0], i[1]]
        temp2 = [i[1]]
        X.append(temp)
        age.append(temp2)
del (X[0])
del (age[0])

out_data = sorted(X, key=lambda x: x[1])
# print(out_data)

y = sorted(age)
str_list = []
str_list.extend([int(x[0]) for x in y])  # Dataset image name as label
# print(str_list)

unique_data = np.unique(y)
# print(unique_data)      # ['03' '04' '05' '06' '07' '08' '09' '10' '11' '12' ... ]

y = np.array(y)
key = np.unique(y)
result = {}  # dictionary
for k in key:
    mask = (y == k)
    y_new = y[mask]
    v = y_new.size
    result[k] = v
# print(result)  # {'03': 1, '04': 1, '05': 1, '06': 3, '07': 2, '08': 1,
mydict = sorted(result.items(), key=lambda item: item[1])
# print(mydict)  # [('03', 1), ('04', 1), ('05', 1), ('08', 1), ('09', 1),
key = [i[0] for i in mydict]
# print(key)
key = list(map(int, key))
# print(key)

val = [i[1] for i in mydict]
# print(val)

dic = {}
for i in range(len(key)):  # the first method
    dic[key[i]] = val[i]
# print(dic)      # {3: 1, 4: 1, 5: 1, 8: 1, 9: 1, 10: 1, 11: 1, ... }

# print(dic.keys())
x1 = list(dic.keys())
x1 = list(map(str, x1))  # Using the list(map(str, x1)) method will return a list, all elements in the list are str type
y1 = list(dic.values())

D1 = int(0.2 * len(unique_data))
D2 = int(0.4 * len(unique_data))
D3 = int(0.6 * len(unique_data))
D4 = int(0.8 * len(unique_data))
D5 = int(1 * len(unique_data))
# print(D1, D2, D3, D4, D5, len(unique_data))  # 14 29 43 58 73 73
# print(dic)
Morph_dic = dic  # {class: number, ... }
print(list(dic.keys()))
Morph_dic_keys = list(dic.keys())  # [class, ... ]
print(list(dic.values()))
Morph_dic_values = list(dic.values())  # [value, ... ]
S1 = list(dic.values())[D1 - 1]
S2 = list(dic.values())[D2 - 1]
S3 = list(dic.values())[D3 - 1]
S4 = list(dic.values())[D4 - 1]
S5 = list(dic.values())[D5 - 1]
print(D1, D2, D3, D4, D5)  # 14  29    43    58    73
print(S1, S2, S3, S4, S5)  # 2   73  1044  1520  2029


def csv_write(csv_path, data):
    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_name', 'label'])
        writer.writerows(data)


# create 2.1
print('creating 2.1')

X_train = []
count = 1
for label in Morph_dic_keys:
    if count <= D1:  # D1, D2, D3, ...
        # Next, the for loop needs to traverse the str_list to output all the indexes,
        # and then fetch elements from X according to the index, and append them to X_train.
        for l in range(0, len(str_list)):
            if label == str_list[l]:
                # print(out_data[l])
                X_train.append(out_data[l])
                csv_write('./ChaLearn15/Train_D1.csv', X_train)
    # else:
    #     print(str_list.index(label))
    # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
    count += 1

X_pre = []
X_final = []
count2 = 1
for label in Morph_dic_keys:
    if count2 <= D1:  # D1, D2, D3, ...
        count2 += 1
        continue  # Go directly to the next for loop
    else:  # after count2 = 15
        # Randomly draw S1 for each label
        for ll in range(0, len(str_list)):
            if label == str_list[ll]:
                # print(out_data[ll])
                X_pre.append(out_data[ll])
                # print(random.sample(X_pre, S1))
                # X_train.append(random.sample(X_pre, S1))
                # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
        # print(random.sample(X_pre, S1))
        X_final.append(random.sample(X_pre, S1))  # S1, S2, S3, ...
        X_pre.clear()
    count2 += 1
# print(X_final)

X_out = []
for i in range(S1):  # S1, S2, S3, ...
    X_out.extend([x[i] for x in X_final])
# X_out.extend([x[1] for x in X_final])
# print(X_out)

X_train.extend(X_out)  # Add all elements of X_out to the end of X_train

csv_write('./datasets/ChaLearn15/Train_D1.csv', X_train)

# kf = KFold(n_splits=10, shuffle=True)
# i = 0
# for train_index, test_index in kf.split(X):
#     X_train = []
#     X_test = []
#     print('split ' + str(i))
#     for j in train_index:
#         X_train.append(X[j])
#     for j in test_index:
#         X_test.append(X[j])
#     # csv_write('./morph_csv/morph_train_%s.csv' % (str(i)), X_train)
#     # csv_write('./morph_csv/morph_val_%s.csv' % (str(i)), X_test)
#     i = i + 1

# print('okk')

# Repeat the above code
# creat 2.2
print('creating 2.2')

X_train2 = []
count = 1
for label in Morph_dic_keys:
    if count <= D2:
        for l in range(0, len(str_list)):
            if label == str_list[l]:
                # print(out_data[l])
                X_train2.append(out_data[l])
                csv_write('./datasets/ChaLearn15/Train_D2.csv', X_train2)
    # else:
    #     print(str_list.index(label))
    # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
    count += 1

X_pre2 = []
X_final2 = []
count2 = 1
for label in Morph_dic_keys:
    if count2 <= D2:  # D1, D2, D3, ...
        count2 += 1
        continue
    else:
        for ll in range(0, len(str_list)):
            if label == str_list[ll]:
                # print(out_data[ll])
                if out_data[ll] in X_train:
                    # print('Duplicate avoided')
                    X_train2.append(out_data[ll])
                else:
                    X_pre2.append(out_data[ll])
                # print(random.sample(X_pre, S1))
                # X_train.append(random.sample(X_pre, S1))
                # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
        # print(random.sample(X_pre, S1))
        X_final2.append(random.sample(X_pre2, S2 - S1))  # S1, S2, S3, ...
        X_pre2.clear()
    count2 += 1
# print(X_final)

X_out2 = []
for i in range(S2 - S1):
    X_out2.extend([x[i] for x in X_final2])
# print(X_out)

X_train2.extend(X_out2)

csv_write('./datasets/ChaLearn15/Train_D2.csv', X_train2)

# creat 2.3
print('creating 2.3')

X_train3 = []
count = 1
for label in Morph_dic_keys:
    if count <= D3:  # D1, D2, D3, ...
        for l in range(0, len(str_list)):
            if label == str_list[l]:
                # print(out_data[l])
                X_train3.append(out_data[l])
                csv_write('./datasets/ChaLearn15/Train_D3.csv', X_train3)
    # else:
    #     print(str_list.index(label))
    # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
    count += 1

X_pre3 = []
X_final3 = []
count2 = 1
for label in Morph_dic_keys:
    if count2 <= D3:  # D1, D2, D3, ...
        count2 += 1
        continue
    else:
        for ll in range(0, len(str_list)):
            if label == str_list[ll]:
                # print(out_data[ll])
                if out_data[ll] in X_train2:
                    # print('Duplicate avoided')
                    X_train3.append(out_data[ll])
                else:
                    X_pre3.append(out_data[ll])
                # print(random.sample(X_pre, S1))
                # X_train.append(random.sample(X_pre, S1))
                # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
        # print(random.sample(X_pre, S1))
        X_final3.append(random.sample(X_pre3, S3 - S2))  # S1, S2, S3, ...
        X_pre3.clear()
    count2 += 1
# print(X_final)

X_out3 = []
for i in range(S3 - S2):
    X_out3.extend([x[i] for x in X_final3])
# print(X_out)

X_train3.extend(X_out3)

csv_write('./datasets/ChaLearn15/Train_D3.csv', X_train3)

# creat 2.4
print('creating 2.4')

X_train4 = []
count = 1
for label in Morph_dic_keys:
    if count <= D4:  # D1, D2, D3, ...
        for l in range(0, len(str_list)):
            if label == str_list[l]:
                # print(out_data[l])
                X_train4.append(out_data[l])
                csv_write('./datasets/ChaLearn15/Train_D4.csv', X_train4)
    # else:
    #     print(str_list.index(label))
    # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
    count += 1

X_pre4 = []
X_final4 = []
count2 = 1
for label in Morph_dic_keys:
    if count2 <= D4:  # D1, D2, D3, ...
        count2 += 1
        continue
    else:
        for ll in range(0, len(str_list)):
            if label == str_list[ll]:
                # print(out_data[ll])
                if out_data[ll] in X_train3:
                    # print('Duplicate avoided')
                    X_train4.append(out_data[ll])
                else:
                    X_pre4.append(out_data[ll])
                # print(random.sample(X_pre, S1))
                # X_train.append(random.sample(X_pre, S1))
                # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
        # print(random.sample(X_pre, S1))
        X_final4.append(random.sample(X_pre4, S4 - S3))  # S1, S2, S3, ...
        X_pre4.clear()
    count2 += 1
# print(X_final)

X_out4 = []
for i in range(S4 - S3):
    X_out4.extend([x[i] for x in X_final4])
# print(X_out)

X_train4.extend(X_out4)

csv_write('./datasets/ChaLearn15/Train_D4.csv', X_train4)

# creat 2.5
print('creating 2.5')

X_train5 = []
count = 1
for label in Morph_dic_keys:
    if count <= D5:  # D1, D2, D3, ...
        for l in range(0, len(str_list)):
            if label == str_list[l]:
                # print(out_data[l])
                X_train5.append(out_data[l])
                csv_write('./datasets/ChaLearn15/Train_D5.csv', X_train5)
    # else:
    #     print(str_list.index(label))
    # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
    count += 1

X_pre5 = []
X_final5 = []
count2 = 1
for label in Morph_dic_keys:
    if count2 <= D5:  # D1, D2, D3, ...
        count2 += 1
        continue
    else:
        for ll in range(0, len(str_list)):
            if label == str_list[ll]:
                # print(out_data[ll])
                if out_data[ll] in X_train4:
                    # print('Duplicate avoided')
                    X_train5.append(out_data[ll])
                else:
                    X_pre5.append(out_data[ll])
                # print(random.sample(X_pre, S1))
                # X_train.append(random.sample(X_pre, S1))
                # csv_write('./morph_csv/morph_train_2.1.csv', X_train)
        # print(random.sample(X_pre, S1))
        X_final5.append(random.sample(X_pre5, S5 - S4))  # S1, S2, S3, ...
        X_pre5.clear()
    count2 += 1
# print(X_final)

X_out5 = []
for i in range(S5 - S4):
    X_out5.extend([x[i] for x in X_final5])
# print(X_out)

X_train5.extend(X_out5)

csv_write('./datasets/ChaLearn15/Train_D5.csv', X_train5)

print('Successfully created all csv files.')

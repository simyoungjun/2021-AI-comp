import numpy as np
import pandas as pd
import os
import time
import scoring as scoring
import pickle
import gzip
from pyarrow import csv
start = time.time()

#####_____________--------------------------------------- File path
train_path = 'C:\\Users\\BB\\Dropbox\\Tclab\\2021 인공지능 온라인 경진대회\\10. 기계시설물 센서 데이터 기반 이상징후 탐지 모델\\대회데이터\\task09_train' # '/DATA/Final_DATA/task09_train'  #
test_path = 'C:\\Users\\BB\\Dropbox\\Tclab\\2021 인공지능 온라인 경진대회\\10. 기계시설물 센서 데이터 기반 이상징후 탐지 모델\\대회데이터\\task09_test'  # '/DATA/Final_DATA/task09_test'   #


#---------------------- Load Train File name
filename_list=[]
for dirName, subdirList, fileList in os.walk(train_path):
    for filename in fileList:
        ext = os.path.splitext(filename)[-1]
        if ext == '.csv':
            filename_list.append([filename])


filename_list = sum(filename_list, [])
print('train_length : ',len(filename_list))
train_length = len(filename_list)



"""
test_file_list = "./sample_submission.csv"
DF = pd.read_csv(test_file_list,usecols = (0,1,2))

#-----------test 파일 불러오는 코드
fn = DF['rand_filename']
fn_val = fn.values
test_filename_list = fn_val.tolist()"""


with gzip.open('./test_filename_list.pickle', 'rb') as f:
    test_filename_list = pickle.load(f)

test_length = len(test_filename_list)

print('test_length : ',test_length)
'''
test_filename_list=[]
for dirName, subdirList, fileList in os.walk(test_path):
    for filename in fileList:
        ext = os.path.splitext(filename)[-1]
        if ext == '.csv':
            test_filename_list.append([filename])


test_filename_list = sum(test_filename_list, [])
print('test_length : ',len(test_filename_list))
test_length = len(test_filename_list)
'''
TA_list = filename_list                     #A의 파일이름 리스트 len =58631 (train 데이터 파일 이름들)
A_list = filename_list + test_filename_list  #A의 파일이름 리스트 len =8744 (test 데이터 파일 이름들)

############## 여기까지까가 train + test의 filename list 생성 : len(67375)인 리스트



filelen = 0
A = [] #np.empty((0,3,2000))

for filename in A_list:
    filelen += 1
    if filelen > train_length:
        file_path = os.path.join(test_path, filename)
    else:
        file_path = os.path.join(train_path,filename)
    # print(file_path)
    #A_unit = pd.read_csv(file_path, usecols=('x', 'y', 'z')) #A_unit.shape = (2000,3)
    A_unit = csv.read_csv(file_path).to_pandas()
    #A_unit = A_unit.iloc[:,1:]

    # print(np.shape)

    #A_unit = A_unit.values
    A_unit = A_unit.T  #A_unit.shape = (3, 2000)

    A.append(A_unit)
    #A = np.append(A, [A_unit], axis=0)
    print("[%d] File loading : %s" %(filelen, filename))

"""



with gzip.open('traindata.pickle', 'rb') as f:
    B = pickle.load(f)

with gzip.open('testdata2.pickle', 'rb') as f:
    C = pickle.load(f)

X = []
X.extend(B)
X.extend(C)

A = np.array(X)
"""

print("================== load complete ==================")
A = np.array(A)
A = A[:,1:,:]
dim_A = A.shape #A.shape = (파일 갯수(=k), 3, 2000)
k = dim_A[0] # = 파일갯수

A1 = abs(np.fft.fft(A[:, 0, :])).T
A2 = abs(np.fft.fft(A[:, 1, :])).T
A3 = abs(np.fft.fft(A[:, 2, :])).T

print("================== fft complete ==================")

tone_mag = np.column_stack([A1[60, :], A2[60, :], A3[60, :]]).T
A1[60, :] = 0
A1[1940, :] = 0
A2[60, :] = 0
A2[1940, :] = 0
A3[60, :] = 0
A3[1940, :] = 0

A1_mag = np.sqrt(sum(A1[:1000, :] ** 2)) / tone_mag[0, :]
A2_mag = np.sqrt(sum(A2[:1000, :] ** 2)) / tone_mag[1, :]
A3_mag = np.sqrt(sum(A3[:1000, :] ** 2)) / tone_mag[2, :]

print("================== mag complete ==================")

A_total = []
A_total.extend(A1[:1000, :])
A_total.extend(A2[:1000, :])
A_total.extend(A3[:1000, :])
A_total = np.array(A_total)

A_total_2 = A_total ** 2
A_total_pw = np.mean(A_total_2, axis=0)
A_total_std = np.std(A_total_2, axis=0)

print("================== create total complete ==================")

feature_vector = np.column_stack([tone_mag.T, A_total_pw.T, A_total_std.T])
feature_vector_new = np.column_stack([tone_mag.T, A1_mag, A2_mag, A3_mag])

print("================== create feature vector complete ==================")

#
sel_feature = feature_vector_new[0:train_length,:]
pts = feature_vector_new[train_length:, :]
std_feature=np.std(sel_feature, axis = 0)

d = feature_vector_new.shape[1] #len(A_total)

c = (4/(d+2)/train_length)**(1/(d+4))

bw = std_feature*c
pa = 0.02


print("================== train start ==================")
# matlab 코드 :
score_01, f1 = scoring.scoring(sel_feature,pts,pa,bw)

temp = list(range(1,len(score_01)+1))

save_data = np.column_stack([temp,test_filename_list,score_01])

df = pd.DataFrame(save_data,columns = ['ID','rand_filename','anomaly_score'])

df.to_csv("./submission.csv",encoding="utf-8",index=False)

print("================== save complete ==================")

print('RunTime : ', time.time()-start)


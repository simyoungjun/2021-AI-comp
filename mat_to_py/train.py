import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
import scoring as scoring

start = time.time()

#####_____________---------------------------------------파일 불러오기
#train 파일 주소

volume_path = 'C:\\Users\\BBK-Zenbook\\Downloads\\task09_test\\task09_test'
train_path = volume_path  #  /DATA/Final_DATA/task01_train
test_path = 'C:\\Users\\BBK-Zenbook\\Downloads\\task09_test\\task09_test'   #/DATA/Final_DATA/task09_test
test_file_list = 'C:\\Users\\BBK-Zenbook\\Downloads\\sample_submission.csv'

#----------------------train 파일 불러오는 코드
filename_list=[]
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(volume_path):
    for filename in fileList:
        filename_list.append([filename])

filename_list = sum(filename_list, [])
# filename_list = filename_list #[:5] 5개 테스트 해보려고 들어가있는듯?
print('len : ',len(filename_list))
train_length = len(filename_list)

# filename_list = filename_list[0:10]


DF = pd.read_csv(test_file_list,usecols = (0,1,2))  # DF: 최종제출 양식(ID,rand_fn,anomaly_score)
# print(DF)
#-----------test 파일 불러오는 코드
fn = DF['rand_filename']
fn_val = fn.values
test_filename_list = fn_val.tolist()
# filename_list =filename_list + fn_val.tolist()
test_length = len(test_filename_list)
# print(filename_list)
print('test_length : ',test_length)

########################################여기까지까가train + test의 filename list 생성 : len(67375)인 리스트


#-----------matlab에 있던 코드 그대로 복붙
power_std = 3
power_mean = [2, 30]
partial_power_ratio = 0.3
DF_index = 0 # 각 파일의 rms를 넣을 때 loc(DF_index,'x') 로 사용함



#----------------------------------train, test 파일 이름 리스트 합침
TA_list= filename_list #A의 파일이름 리스트 len =58631 (train 데이터 파일 이름들)
A_list = filename_list+test_filename_list  #A의 파일이름 리스트 len =8744 (test 데이터 파일 이름들)
# print("---------train 파일개수", len(filename_list))
A = np.empty((0,3,2000))
A_list=A_list[0:100]  #################################################### test 를 위해 잠깐 해놓은것 나중에 지워야함

filelen = 0
#--------------matlab에서의 A,TA( 2000x3x67375만드는 과정) np.array
for filename in A_list:
    filelen+=1

    file_path = os.path.join(volume_path,filename)
    # print(file_path)
    A_unit = pd.read_csv(file_path, usecols=('x', 'y', 'z')) #A_unit.shape = (2000,3)
    # print(np.shape)

    A_unit = A_unit.values
    A_unit = np.transpose(A_unit) #A_unit.shape = (3, 2000)


    A = np.append(A,[A_unit],axis=0)
    print("[%d] File loading : %s" %(filelen, filename))


#np.save('./data.npy',A)
#np.load('./data.npy',allow_pickle=True)
dim_A = A.shape #A.shape = (파일 갯수(=k), 3, 2000)
k = dim_A[0] # = 파일갯수


# A1 = np.empty((0,2000))
# A2 = np.empty((0,2000))
# A3 = np.empty((0,2000))
A_xyz_list = [] # 나중에 A1,A2,A3(=매트랩에서 변수이름)를 원소로 갖는 리스트가 됨==[A1,A2,A3]

fs = 2000
t = np.arange(0,1,1/fs)

for x_y_z in range(3):
    A_xyz = np.empty((0, 2000)) # == for문의 x_y_z에 따라 매트랩에서의 A1,A2,A3이 됨
    for i in A[:,x_y_z,:]: #i는 67375 중에 한개 [1,2000]
        # print('i : ',i.shape) #(2000,)
        # print(i)
        fft = np.fft.fft(i) / i.size
        fft_magnitude = abs(fft) #A1, A2, A3의 유닛이 됨
        A_xyz = np.append(A_xyz,[abs(np.fft.fft(i))/i.size],axis=0)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        #
        # print('A1.shape : ', A1.shape)
        # plt.subplot(2, 1, 1)
        # plt.plot(t, fft_magnitude )
        # plt.grid()
        #
        # plt.subplot(2, 1, 2)
        # plt.stem(fft_magnitude)
        # plt.ylim(0, 2.5)
        # plt.grid()
        #
        # plt.show()
    A_xyz_list.append(A_xyz) # A1, A2, A3을 합친 것 == [A1,A2,A3], 밑의 tone_mag 편하게 하기 위해 만듬
print('len(A_xyz_list', len(A_xyz_list))
print('A_xyz_list[0].shape : ',A_xyz_list[0].shape)

##################
# A1 == A_xyz_list[0]
# A1 == A_xyz_list[1]
# A1 == A_xyz_list[2] 이다.
#############

#----------------- matlab의 %feature 부분
tone_mag = np.empty((0,k)) # k는 파일의 갯수?

A_total = np.empty((0,k))
# A_total = np.empty((0,1000,k))
A_123_mag=[]
n=0
for A_xyz in A_xyz_list: #A_xyz 는 for문에 의해 A1,A2,A3가 됨
    A_xyz = np.transpose(A_xyz) #A_xyz.shape를 (2000,(파일갯수=k)로 바꿔줌)
    print('A1(61,:)', A_xyz[60 ,:].shape) #A_xyz.shpae = (67375(=파일개수),2000)
    A_xyz_f = abs(A_xyz[60, :])
    tone_mag = np.append(tone_mag, [A_xyz_f], axis=0)

    A_xyz[60,:] = 0 #--> matlab에서의 인덱스 61 == 파이썬의 60
    A_xyz[1940,:] = 0

    A_123_mag.append(np.transpose((np.sqrt(sum(A_xyz[0:1000,:])/tone_mag[n])).reshape(-1,1))) #shape : (1,K(파일개수))
    print('A_123.shape',A_123_mag[n].shape) # SHAPE = (1,K)
    n+=1
#A_total 만드는 부분 맞게 했는지 잘 모르겠습니다. matlab에서 보면 3차원 같은데 shape은 2차원으로 나와서..
    A_total = np.append(A_total, A_xyz[0:1000,:],axis = 0) #파이썬에서의 인덱스 맞는지? matlab과 비교 shape=(3000,k)
   #A_total = np.append(A_total, [A_xyz[0:1000,:]],axis = 0) #파이썬에서의 인덱스 맞는지? matlab과 비교 shape =(3,1000,k)
#위의 두 코드 중 어떤 게 맞는 건지?


A_total_pw = np.mean(A_total**2,axis =0)
# A_total_pw = np.empty((0))
# A_total_std = np.empty((0))
# A_total = np.transpose(A_total) #shape(k,3000)으로 바뀜
# for A_total_unit in A_total:
#     A_total_pw = np.append(A_total_pw, [mean(A_total_unit)])
print('tone_mag.shape : ', tone_mag.shape) # shape = (3,파일개수(=67375))
print('A_total : ', A_total.shape)


########여기맞게 했는지? 매트랩에서 A_total_pw=mean(A_total.^2)'
A_total_pw = np.mean(A_total**2,axis = 0).reshape(-1,1)
A_total_std = np.std(A_total**2,axis = 0).reshape(-1,1)
print('A_total_pw.shape : ',A_total_pw.shape)
print('A_total_std.shape : ',A_total_std.shape)

# print('np.treanspose(tone_mag).shape', np.transpose(tone_mag).shape)
feature_vector=np.hstack((np.transpose(tone_mag),A_total_pw,A_total_std))
feature_vector_new = np.hstack((np.transpose(tone_mag), np.transpose(A_123_mag[0]),np.transpose(A_123_mag[0]),np.transpose(A_123_mag[0])))
#feature_vector.shape = (k, 5)

# sel_feature=feature_vector[0:train_length,:]
sel_feature=feature_vector_new
pts = feature_vector_new[:,:] #
#pts =  feature_vector_new[train_length-test_length:,:]
std_feature=np.std(sel_feature, axis = 0)

d=len(A_total)

c=(4/(d+2)/train_length)**(1/(d+4))

bw= std_feature*c


pa=0.0065 #0.0423

# matlab 코드 :
score_01_1,f1 = scoring.scoring(sel_feature,pts,pa,bw)

pa_ref=0.007

# matlab 코드 :
score_01_2,f2 = scoring.scoring(sel_feature ,pts,pa_ref,bw)

diff_scoring=score_01_1-score_01_2
score_01 = score_01_1
stats.norm(diff_scoring)

a = np.sort(score_01_1.T)
b = np.sort(score_01_2.T)
print(a)
plt.plot(a.T,'g--')
plt.plot(b.T,'r')
#plt.show()


temp = list(range(1,len(score_01)+1))


#savedata =np.column_stack([temp,test_filename_list,score_01])
savedata =np.column_stack([temp,test_filename_list[:100],score_01]) ######################## test 를 위해 잠깐 해놓은것, 나중에 바로 위 코드 써야함

df = pd.DataFrame(savedata,columns = ['ID','rand_filename','anomaly_score'])

df.to_csv("submission.csv",encoding="utf-8",index=False)
print("save complete")

print('걸린 시간 : ',time.time()-start)


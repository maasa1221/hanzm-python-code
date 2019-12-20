from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

df0=pd.read_csv("data/new/fujitaRP1.csv",index_col=1)
df2=pd.read_csv("data/new/koyamaRP1.csv",index_col=1)
df3=pd.read_csv("data/new/koyanagiRP1.csv",index_col=1)
df4=pd.read_csv("data/new/nimuraRP1.csv",index_col=1)
df5=pd.read_csv("data/new/suzukiRP1.csv",index_col=1)
df6_c=pd.read_csv("data/new/64058436RP1.csv",index_col=1)
df7_c=pd.read_csv("data/new/79122114RP1.csv",index_col=1)
df8_c=pd.read_csv("data/new/52193348RP1.csv",index_col=1)
df9_c=pd.read_csv("data/new/19099510RP1.csv",index_col=1)
df10_c=pd.read_csv("data/new/19110622RP1.csv",index_col=1)
df11_c=pd.read_csv("data/new/19630516RP1.csv",index_col=1)
df12_c=pd.read_csv("data/new/82304270RP1.csv",index_col=1)
df13_c=pd.read_csv("data/new/15662472RP1.csv",index_col=1)
df14_c=pd.read_csv("data/new/19600594RP1.csv",index_col=1)
df15_c=pd.read_csv("data/new/73209319RP1.csv",index_col=1)
df16_c=pd.read_csv("data/new/18611860R_0.csv",index_col=1)
df17_c=pd.read_csv("data/new/19619780R_0.csv",index_col=1)
df18_c=pd.read_csv("data/new/19636280R0.csv",index_col=1)
df19_c=pd.read_csv("data/new/19639492R_0.csv",index_col=1)
df20_c=pd.read_csv("data/new/67090028R_0.csv",index_col=1)
df21_c=pd.read_csv("data/new/81160006R_0.csv",index_col=1)
df22=pd.read_csv("data/masaru_0.csv",index_col=1)
df23=pd.read_csv("data/etukoRP1.csv",index_col=1)
df24=pd.read_csv("data/turu22.csv",index_col=1)
#df25=pd.read_csv("data/turu23.csv",index_col=1)
df26=pd.read_csv("data/turu20.csv",index_col=1)
df27=pd.read_csv("data/turu19.csv",index_col=1)
df28=pd.read_csv("data/turu18.csv",index_col=1)
#df29=pd.read_csv("data/turu15.csv",index_col=1)
#df30=pd.read_csv("data/turu14_0.csv",index_col=1)
df31=pd.read_csv("data/turu11.csv",index_col=1)
df32=pd.read_csv("data/turu10.csv",index_col=1)
df33=pd.read_csv("data/turu9_0.csv",index_col=1)
#df34=pd.read_csv("data/turu1.csv",index_col=1)
df35_c=pd.read_csv("data/18695075.csv",index_col=1)
df36_c=pd.read_csv("data/19025608.csv",index_col=1)
df37_c=pd.read_csv("data/19633531.csv",index_col=1)
df38_c=pd.read_csv("data/19638894.csv",index_col=1)
df39_c=pd.read_csv("data/88621754.csv",index_col=1)
df40_c=pd.read_csv("data/86140331_c.csv",index_col=1)
df41_c=pd.read_csv("data/19639054_c.csv",index_col=1)
df42_c=pd.read_csv("data/19629494_c.csv",index_col=1)
df43_c=pd.read_csv("data/19169481_c.csv",index_col=1)
df44_c=pd.read_csv("data/17167931_c.csv",index_col=1)
df45_c=pd.read_csv("data/16651981_c.csv",index_col=1)
df46_c=pd.read_csv("data/16639522_c.csv",index_col=1)
df47_c=pd.read_csv("data/15620160_c.csv",index_col=1)
df48=pd.read_csv("data/74123588.csv",index_col=1)
df49=pd.read_csv("data/73043426.csv",index_col=1)
df50=pd.read_csv("data/19653527.csv",index_col=1)
df51=pd.read_csv("data/19652314.csv",index_col=1)
df52=pd.read_csv("data/19651592.csv",index_col=1)
df53=pd.read_csv("data/19648231.csv",index_col=1)
df54=pd.read_csv("data/19630892.csv",index_col=1)
df55=pd.read_csv("data/19140706.csv",index_col=1)
df56=pd.read_csv("data/18693077.csv",index_col=1)
df57=pd.read_csv("data/18642391.csv",index_col=1)
df58_c=pd.read_csv("data/18684050_c.csv",index_col=1)
df59_c=pd.read_csv("data/19627664_c.csv",index_col=1)
df60_c=pd.read_csv("data/19639606_c.csv",index_col=1)
df61_c=pd.read_csv("data/19641229_c.csv",index_col=1)
df62_c=pd.read_csv("data/19648874_c.csv",index_col=1)
df63_c=pd.read_csv("data/19650334_c.csv",index_col=1)
df64_c=pd.read_csv("data/19662018_c.csv",index_col=1)
df65_c=pd.read_csv("data/19680864_c.csv",index_col=1)
df66_c=pd.read_csv("data/86141143_c.csv",index_col=1)
df67_c=pd.read_csv("data/87175907_c.csv",index_col=1)
df68=pd.read_csv("data/15012763.csv",index_col=1)
df69=pd.read_csv("data/16605773.csv",index_col=1)
df70=pd.read_csv("data/19632168.csv",index_col=1)
df71=pd.read_csv("data/19674871.csv",index_col=1)
df72=pd.read_csv("data/19684263.csv",index_col=1)
df73=pd.read_csv("data/61176723.csv",index_col=1)
df74=pd.read_csv("data/79303810.csv",index_col=1)
df75=pd.read_csv("data/85226517.csv",index_col=1)
f_Recall=0
f_Specificity=0
f_AUC=0
f_Accuracy=0
s_Recall=0
s_Specificity=0
s_AUC=0
s_Accuracy=0
tree_list_box=[]
score_list_box=[]
for z in range(66):
    n=1
    nnn=1
    # parameters
    N = 128 # data number
    dt = 1/60 # data step [s]
    train_data_contena=[]
    train_target_contena=[]
    normal_counter=32 #21  20 23
    counter=0
    human_list=[df0,df2,df5,df3,df4,df22,df23,df24,df26,df27,df28,df31,df32,df33,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df68,df69,df70,df71,df72,df73,df74,df75,df6_c,df7_c,df8_c,df9_c,df10_c,df11_c,df12_c,df13_c,df14_c,df15_c,df35_c,df36_c,df37_c,df38_c,df39_c,df16_c,df17_c,df19_c,df18_c,df20_c,df21_c,df40_c,df41_c,df42_c,df43_c,df44_c,df45_c,df46_c,df47_c,df58_c,df59_c,df60_c,df61_c,df62_c,df63_c,df64_c,df65_c,df66_c,df67_c]
    pred_goukei=0
    a=64
    mm = MinMaxScaler()
    def ffts(array,number_tree):
        data_contena=[]
        target_contena=[]
        y = array.iloc[:,number_tree] 
        for i in range(len(y)-250,len(y),20):#512の範囲を1ストライド
            y1=y.iloc[i-128:i]#入力データの128フェーズへの切り出し
            y1=np.array(y1)#　np.ndarray化
            data_regista=np.zeros((nnn,a))
            yf = fft(y1)/(N/2)#離散フーリエ変換&規格化<-
            data_regista=yf[1:65] #5次元分レジスタに格納
            #data_regista=scipy.stats.zscore(data_regista)#<=======================
            #print(data_regista)
            data_regista2=data_regista.reshape(a*nnn)#次元変形
            data_contena.append(data_regista2)#コンテナに追加
            #print(np.array(data_contena).shape)
            #print(np.abs(data_contena).shape)
            if counter>=normal_counter:            
                target_contena.append(0)
            else:
                target_contena.append(1)
        return data_contena, target_contena

    first_score=0
    first_vec=[]
    second=[]
    first_tree_num=0
    fig=plt.figure(figsize=(20,20))
    final_contena=[]
    final_data_contena=[]
    for j in range(z,z+1):
        n=1
        nnn=1
        # parameters
        N = 128 # data number
        dt = 1/60 # data step [s]
        train_data_contena=[]
        train_box=[]
        train_target_contena=[]
        target_box=[]
        normal_counter=24 #21
        counter=0
        human_list=[df0,df2,df5,df3,df4,df22,df23,df24,df26,df27,df28,df31,df32,df33,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df68,df69,df70,df71,df72,df73,df74,df75,df6_c,df7_c,df8_c,df9_c,df10_c,df11_c,df12_c,df13_c,df14_c,df15_c,df35_c,df36_c,df37_c,df38_c,df39_c,df16_c,df17_c,df19_c,df18_c,df20_c,df21_c,df40_c,df41_c,df42_c,df43_c,df44_c,df45_c,df46_c,df47_c,df58_c,df59_c,df60_c,df61_c,df62_c,df63_c,df64_c,df65_c,df66_c,df67_c]
        a=64
        for i in human_list:
            train_box=[]
            target_box=[]
            train_box,target_box=ffts(i,j)
            if counter==0:
                reshape_base=np.array(target_box).size
            #print(np.array(train_box).shape)
            train_data_contena.append(np.array(train_box))
            train_target_contena.append(np.array(target_box))
            counter+=1
        train_data_contena=np.array(np.abs(train_data_contena))
        train_data_contena=train_data_contena.reshape(71*reshape_base,1*a)
        train_target_contena=np.array(train_target_contena).reshape(71*reshape_base)
        train_data_contena2= mm.fit_transform(train_data_contena)
        final_contena.append(np.array(train_data_contena2))
        final_data_contena.append(np.array(train_target_contena))
    final_contena=np.array(final_contena).reshape(923,64*1)
    clf = RandomForestClassifier()#<-------------------------------------------------------------------------------------------------------------------------
    pca = PCA(n_components=3)  
    X=pca.fit_transform(final_contena)
    clf2=clf.fit(final_contena,  train_target_contena)
    print(X.shape)
    stratifiedkfold = StratifiedKFold(n_splits=6)
    pred_score=cross_val_score(clf, final_contena, train_target_contena,cv=stratifiedkfold)#各スコア¥
    pred_score=pred_score.mean()
    pred_vec=cross_val_predict(clf, final_contena, train_target_contena,cv=stratifiedkfold)#各各の判断結果配列
    first_vec=pred_vec
    print( pred_score)
first=confusion_matrix(train_target_contena, pred_vec)
f_Recall+=first[0][0]/(first[0][0]+first[0][1])
f_Specificity+=first[1][1]/(first[1][0]+first[1][1])
f_AUC+=roc_auc_score(train_target_contena, first_vec)
f_Accuracy+=(first[1][1]+first[0][0])/(first[0][0]+first[0][1]+first[1][0]+first[1][1])
print(f_Recall)
print(f_Specificity)
print(f_AUC)
print(f_Accuracy)
print(first)
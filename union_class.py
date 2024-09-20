import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split,Subset
from torchvision.datasets import ImageFolder
from sklearn.metrics import  confusion_matrix
from torch.utils.data.sampler import WeightedRandomSampler
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,CIFAR100
import pickle

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 隨機裁剪
    transforms.RandomHorizontalFlip(),     # 隨機水平翻轉
    transforms.RandomRotation(15),         # 隨機旋轉
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的均值和標準差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
model_algo="resnet20"
import sys

# 添加自定义模块的目录到模块搜索路径
sys.path.append('/home/pcdm/Desktop/mjpaper/cifar100')

# 现在可以导入你的自定义模块
from MJPytorch import *

import os
os.chdir(r'/home/pcdm/Desktop/mjpaper/cifar100')

# CIFAR-100 資料集
trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainset=CifarDataset(trainset)


validset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
validset=CifarDataset(validset)
# create data loaders
batch_size =32 # larger numbers lead to CUDA running out of memory
train_dl = DataLoader(trainset,shuffle=True, batch_size=batch_size)
valid_dl = DataLoader(validset,shuffle=True, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
# model_0=model_train(model_algo,train_dl,valid_dl,"0",100)
model_0 = torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_{model_algo}_0.pth')
train_size=len(trainset)
valid_size=len(validset)

train_loss,train_accu,confusion_matrix_train=evaluate_model(model_0,train_dl,train_size,'0',mode='eval')

valid_loss,valid_accu,confusion_matrix_valid=evaluate_model(model_0,valid_dl,valid_size,'0',mode='eval')
r=np.diag(confusion_matrix_train)/confusion_matrix_train.sum(0)
p=np.diag(confusion_matrix_train)/confusion_matrix_train.sum(1)
each_accu=p
each_accu=each_accu.sort_values(ascending=True)
each_accu
each_f1=2*p*r/(p+r)

def generate_union_model(split_mode):

    if(split_mode[0]=='classf1'):
        targetF=[x  for x in each_f1.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    elif(split_mode[0]=='classaccu'):
        targetF=[x  for x in each_accu.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    else:
        split=split_mode


    with open(f'/home/pcdm/Desktop/mjpaper/cifar100/res20_classf1_indices_{split_mode[1]}.pkl', 'rb') as f:
        data = pickle.load(f)
        indexF = data['indexF']
        indexT = data['indexT']
        indexF_v = data['indexF_v']
        indexT_v = data['indexT_v']

    def turn_flag(index,data_dl):
        for i in index:
            data_dl.dataset.update_flag(i)

    turn_flag(indexF,train_dl)
    turn_flag(indexF_v,valid_dl)

    Fdataset=Subset(trainset, indexF)
    Tdataset=Subset(trainset, indexT)

    Fdl=DataLoader(Fdataset, shuffle=True, batch_size=batch_size)
    Tdl=DataLoader(Tdataset, shuffle=True, batch_size=batch_size)

    Fdataset_v=Subset(validset, indexF_v)
    Tdataset_v=Subset(validset, indexT_v)

    Fdl_v=DataLoader(Fdataset_v, shuffle=True, batch_size=batch_size)
    Tdl_v=DataLoader(Tdataset_v, shuffle=True, batch_size=batch_size)
    # model_decision=model_train(model_algo,train_dl,valid_dl,"decision",100,model_0=model_0,root=split_mode)
    model_decision = torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_resnet20_decision_{split_mode}.pth')
    print("Load model decision OK.")

    #訓練模型T、F、decision
    # model_T=model_train(model_algo,Tdl,Tdl_v,"T",100,model_0=model_0,root=split_mode)
    model_T=torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_resnet20_T_{split_mode}.pth')
    print("Load model_T OK.")

    #訓練模型T、F、decision
    # model_F=model_train(model_algo,Fdl,Fdl_v,"F",100,model_0=model_0,root=split_mode)
    model_F=torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_resnet20_F_{split_mode}.pth')
    print("Load model_F OK.")

    decision_loss,decision_accu,confusion_matrix_decision_train=evaluate_model(model_decision,train_dl,len(trainset),'decision',mode='eval')
    confusion_matrix_decision_train= confusion_matrix_decision_train.rename(columns={"0":"False","1":"True"}, index={"0":"False","1":"True"})
    decision_valid_loss,decision_valid_accu,confusion_matrix_decision_valid=evaluate_model(model_decision,valid_dl,len(validset),'decision',mode='eval')
    confusion_matrix_decision_valid= confusion_matrix_decision_valid.rename(columns={0:"False",1:"True"}, index={0:"False",1:"True"})
    T_loss,T_accu,confusion_matrix_T_train=evaluate_model(model_T,Tdl,len(Tdataset),'T',mode='eval')
    F_loss,F_accu,confusion_matrix_F_train=evaluate_model(model_F,Fdl,len(Fdataset),'F',mode='eval')
    T_valid_loss,T_valid_accu,confusion_matrix_T_valid=evaluate_model(model_T,Tdl_v,len(Tdataset_v),'T',mode='eval')
    F_valid_loss,F_valid_accu,confusion_matrix_F_valid=evaluate_model(model_F,Fdl_v,len(Fdataset_v),'F',mode='eval')
    total_train_loss_d,total_train_accu_d,confusion_matrix_total_train,sum=total_model_evaluate_notdecision(train_dl,train_size,model_0,model_T,model_F)
    total_valid_loss_d,total_valid_accu_d,confusion_matrix_total_valid_d,sum=total_model_evaluate_notdecision(valid_dl,valid_size,model_0,model_T,model_F)
    total_train_loss,total_train_accu,confusion_matrix_total_train,sum_F,sum_T=total_model_evaluate(train_dl,train_size,model_0,model_T,model_F,model_decision,decision_mode='model')
    total_valid_loss,total_valid_accu,confusion_matrix_total_valid,sum_F,sum_T=total_model_evaluate(valid_dl,valid_size,model_0,model_T,model_F,model_decision,decision_mode='model')

    train_losses=[train_loss,decision_loss,T_loss,F_loss,total_train_loss,total_train_loss_d]
    train_accuracies=[train_accu,decision_accu,T_accu,F_accu,total_train_accu,total_train_accu_d]
    valid_losses=[valid_loss,decision_valid_loss,T_valid_loss,F_valid_loss,total_valid_loss,total_valid_loss_d]
    valid_accuracies=[valid_accu,decision_valid_accu,T_valid_accu,F_valid_accu,total_valid_accu,total_valid_accu_d]
    model_algos=['Model_0','Model_Decision',"Model_T","Model_F","Total_Model",'Total_Model_with_perfect_decision']
    data = {
        'Model': model_algos,
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Valid Loss': valid_losses,
        'Valid Accuracy': valid_accuracies
    }
    df = pd.DataFrame(data)
    df['split_mode'] =str(split_mode)
    print(f'split_mode = {split}')
    print(df)

    df_results=show_model_evaluate(train_dl,train_size,model_0,model_T,model_F,model_decision)

    df_filter=(df_results[(df_results['Decision']!=df_results['flag']) ])
    class_counts = df_filter['Target'].value_counts()
    
    plt.figure(figsize=(10, 6))

    plt.hist(df_filter['Target'], bins=68, color='skyblue', edgecolor='black')
  
    plt.title(f'Histogram of Errors in Multi-Model Selection(n={split_mode[1]})')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    plt.savefig(f'histogram_n={split_mode[1]}.png')  # 可以指定文件路径和文件格式，如 .png, .jpg, .pdf 等

    plt.close()

    df_list=[]
    for union in range(5,25,5):
        targetF_2=[x  for x in class_counts.nlargest(union).index]       
        indexF_u=[]
        indexT_u=[]

        if(split_mode[0]=='classf1'):
            targetF=[x  for x in each_f1.nsmallest(split_mode[1]).index.map(trainset.classess)]
            split=(split_mode[0],targetF,targetF_2)
        elif(split_mode[0]=='classaccu'):
            targetF=[x  for x in each_accu.nsmallest(split_mode[1]).index.map(trainset.classess)]
            split=(split_mode[0],targetF,targetF_2)
        else:
            split=split_mode
        
        #將資料切分成true and false
        def split_data(model_0,data_dl,split_mode):
            model_0.eval()
            indexF=[]
            indexT=[]
            with torch.no_grad():
                for (data,target,_,idx) in data_dl:
                    data,target=data.cuda(),target.cuda()
                    out = model_0(data)
                    _, y_pred_tag = torch.max(out, dim = 1) 
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    loss = criterion(out, target)
                    softmax = torch.softmax(out, dim=1)

                    for idx,loss,t,pred ,softmax in zip(idx,loss,target,y_pred_tag,softmax):
                        if(split_mode[0]=='loss'):    
                            if(loss>split_mode[1]):
                                indexF.append(idx.cpu().numpy().item())  
                                data_dl.dataset.update_flag(idx)
                            else:
                                indexT.append(idx.cpu().numpy().item())  

                        elif(split_mode=='TandF'):
                            if(t!=pred):
                                indexF.append(idx.cpu().numpy().item())  
                                data_dl.dataset.update_flag(idx)
                            else:
                                indexT.append(idx.cpu().numpy().item())
                            
                        
                        elif(split_mode[0]=='softmax'):
                        
                            if(softmax.max()<split_mode[1]):
                                indexF.append(idx.cpu().numpy().item())  
                                data_dl.dataset.update_flag(idx)
                            else:
                                indexT.append(idx.cpu().numpy().item())
                                # if(softmax.max()<split_mode[2]):
                                #     indexF.append(idx.cpu().numpy().item())

                        elif(split_mode[0]=='classaccu'or split_mode[0]=='classf1'):
                            if(t in split_mode[1]):
                                indexF.append(idx.cpu().numpy().item())  
                                data_dl.dataset.update_flag(idx)
                                if(t in split_mode[2]):
                                    indexT.append(idx.cpu().numpy().item())    
                            else:
                                indexT.append(idx.cpu().numpy().item())
                                if(t in split_mode[2]):
                                    indexF.append(idx.cpu().numpy().item())             
                                    
                torch.cuda.empty_cache() 
            return indexF,indexT

           #for resnet use
        def model_train(model_algo,train_dl,valid_dl,data_name,epochs,model_0=None,root=None):

            if(data_name=="decision"):
                # model=model_create(model_algo,data_name,2)
                model=copy.deepcopy(model_0)
                model.linear=nn.Linear(model.linear.in_features,2)

            elif(data_name=='T'or data_name=='F'):
                model=copy.deepcopy(model_0)
                model.linear=nn.Linear(model.linear.in_features,len(train_dl.dataset.dataset.classes))

            else:
                model=model_create(model_algo,data_name,len(train_dl.dataset.classes))

            
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss() 
            model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name, model_algo,root)

            torch.cuda.empty_cache() 
            return model

        indexF_u,indexT_u=split_data(model_0,train_dl,split)

        Fdataset_u=Subset(trainset, indexF_u)
        Tdataset_u=Subset(trainset, indexT_u)

        Fdl_u=DataLoader(Fdataset_u, shuffle=True, batch_size=batch_size)
        Tdl_u=DataLoader(Tdataset_u, shuffle=True, batch_size=batch_size)


        indexF_v_u=[]
        indexT_v_u=[]

        indexF_v_u,indexT_v_u=split_data(model_0,valid_dl,split)
        Fdataset_v_u=Subset(validset, indexF_v_u)
        Tdataset_v_u=Subset(validset, indexT_v_u)

        Fdl_v_u=DataLoader(Fdataset_v_u, shuffle=True, batch_size=batch_size)
        Tdl_v_u=DataLoader(Tdataset_v_u, shuffle=True, batch_size=batch_size)
        model_name=f"{split_mode}_union_num={union}"
        print(split)

        data = {'indexF_u': indexF_u, 'indexT_u': indexT_u,'indexF_v_u': indexF_v_u, 'indexT_v_u': indexT_v_u}
        with open(f'{split_mode[0]}_indices_{split_mode[1]}_u_{union}.pkl', 'wb') as f:
            pickle.dump(data, f)


        #訓練模型T、F、decision
        if len(indexT_u)!=len(indexT):
            print("Training union model_T !!!!")
            model_T_u=model_train(model_algo,Tdl_u,Tdl_v_u,"T",100,model_0=model_0,root=model_name)
            # model_T_u=torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_alexnet_T_{split_mode}_union_num={union}.pth')
            
        else:
            model_T_u=copy.deepcopy(model_T)
     
        #訓練模型T、F、decision
        if len(indexF_u)!=len(indexF):
            print("Training union model_F !!!!")
            model_F_u=model_train(model_algo,Fdl_u,Fdl_v_u,"F",100,model_0=model_F,root=model_name)
            # model_F_u=torch.load(f'/home/pcdm/Desktop/mjpaper/cifar100/model_alexnet_F_{split_mode}_union_num={union}.pth')

        else:
            model_F_u=copy.deepcopy(model_F)

        print("Train finish !!!!")

        T_loss_u,T_accu_u,confusion_matrix_T_train=evaluate_model(model_T_u,Tdl_u,len(Tdataset_u),'T',mode='eval')
        F_loss_u,F_accu_u,confusion_matrix_F_train=evaluate_model(model_F_u,Fdl_u,len(Fdataset_u),'F',mode='eval')
        T_valid_loss_u,T_valid_accu_u,confusion_matrix_T_valid=evaluate_model(model_T_u,Tdl_v_u,len(Tdataset_v_u),'T',mode='eval')
        F_valid_loss_u,F_valid_accu_u,confusion_matrix_F_valid=evaluate_model(model_F_u,Fdl_v_u,len(Fdataset_v_u),'F',mode='eval')
        total_train_loss_d_u,total_train_accu_d_u,confusion_matrix_total_train,sum=total_model_evaluate_notdecision(train_dl,train_size,model_0,model_T_u,model_F_u)
        total_valid_loss_d_u,total_valid_accu_d_u,confusion_matrix_total_valid_d,sum=total_model_evaluate_notdecision(valid_dl,valid_size,model_0,model_T_u,model_F_u)
        total_train_loss_u,total_train_accu_u,confusion_matrix_total_train,sum_F,sum_T=total_model_evaluate(train_dl,train_size,model_0,model_T_u,model_F_u,model_decision,decision_mode='model')
        total_valid_loss_u,total_valid_accu_u,confusion_matrix_total_valid,sum_F,sum_T=total_model_evaluate(valid_dl,valid_size,model_0,model_T_u,model_F_u,model_decision,decision_mode='model')

        train_losses_u=[train_loss,decision_loss,T_loss_u,F_loss_u,total_train_loss_u,total_train_loss_d_u]
        train_accuracies_u=[train_accu,decision_accu,T_accu_u,F_accu_u,total_train_accu_u,total_train_accu_d_u]
        valid_losses_u=[valid_loss,decision_valid_loss,T_valid_loss_u,F_valid_loss_u,total_valid_loss_u,total_valid_loss_d_u]
        valid_accuracies_u=[valid_accu,decision_valid_accu,T_valid_accu_u,F_valid_accu_u,total_valid_accu_u,total_valid_accu_d_u]
        model_algos=['Model_0','Model_Decision',f"Model_T_{union}","Model_F","Total_Model",'Total_Model_with_perfect_decision']
        data_u = {
            'Model': model_algos,
            'Train Loss': train_losses_u,
            'Train Accuracy': train_accuracies_u,
            'Valid Loss': valid_losses_u,
            'Valid Accuracy': valid_accuracies_u
        }
        df_u = pd.DataFrame(data_u)
        print(df_u) 
        df_list.append(df_u)
        empty_row=pd.DataFrame(['']*len(df),columns=[''])
        df_list.append(empty_row)
    
    combined_df=pd.concat(df_list,axis=1)

    return df,combined_df

# i=10    

# while i<100:
i=30
split_mode=('classaccu',i)
write_header=True
df,combined_df=generate_union_model(split_mode)


df.to_csv(f'rs20_output_ciar100__{i}_union.csv', mode='a', header=write_header, index=False )
write_header = False
empty_df = pd.DataFrame([[]])
empty_df.to_csv(f'rs20_output_ciar100_f1_{i}_union.csv', mode='a', header=False, index=False)

write_header=True
combined_df.to_csv(f'rs20_output_ciar100_f1_{i}_union.csv', mode='a', header=write_header, index=False )
write_header = False
empty_df = pd.DataFrame([[]])
empty_df.to_csv(f'rs20_output_ciar100_f1_{i}_union.csv', mode='a', header=False, index=False)
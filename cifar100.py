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

import sys
sys.path.append(r'/home/pcdm/Desktop/mjpaper/cifar100')
from MJPytorch import *
import os
os.chdir(r'/home/pcdm/Desktop/mjpaper/cifar100')

def training_loop(split_mode):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),     
        transforms.RandomRotation(15),         
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    model_algo="vgg16"
    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainset=CifarDataset(trainset)


    validset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    validset=CifarDataset(validset)

    batch_size =32 # larger numbers lead to CUDA running out of memory
    train_dl = DataLoader(trainset,shuffle=True, batch_size=batch_size)
    valid_dl = DataLoader(validset,shuffle=True, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    # model_0=model_train(model_algo,train_dl,valid_dl,"0",100)
    model_0 = torch.load(rf'/home/pcdm/Desktop/mjpaper/cifar100/model_{model_algo}_0.pth')
    train_size=len(trainset)
    valid_size=len(validset)

    train_loss,train_accu,confusion_matrix_train=evaluate_model(model_0,train_dl,train_size,'0',mode='eval')

    valid_loss,valid_accu,confusion_matrix_valid=evaluate_model(model_0,valid_dl,valid_size,'0',mode='eval')

    r=np.diag(confusion_matrix_train)/confusion_matrix_train.sum(0)
    p=np.diag(confusion_matrix_train)/confusion_matrix_train.sum(1)
   
    each_accu=p
    each_accu=each_accu.sort_values(ascending=True)

    each_f1=2*p*r/(p+r)

    #將資料切分後建立dataset與dataloader
    indexF=[]
    indexT=[]

    if(split_mode[0]=='classf1'):
        targetF=[x  for x in each_f1.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    elif(split_mode[0]=='classaccu'):
        targetF=[x  for x in each_accu.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    else:
        split=split_mode


    indexF,indexT=split_data(model_0,train_dl,split)

    Fdataset=Subset(trainset, indexF)
    Tdataset=Subset(trainset, indexT)

    Fdl=DataLoader(Fdataset, shuffle=True, batch_size=batch_size)
    Tdl=DataLoader(Tdataset, shuffle=True, batch_size=batch_size)

    len(indexF),len(indexT)
    indexF_v=[]
    indexT_v=[]

    indexF_v,indexT_v=split_data(model_0,valid_dl,split)

    # #for resnet use
    # def model_train(model_algo,train_dl,valid_dl,data_name,epochs,model_0=None,root=None):

    #     if(data_name=="decision"):
    #         # model=model_create(model_algo,data_name,2)
    #         model=copy.deepcopy(model_0)
    #         model.linear=nn.Linear(model.linear.in_features,2)

    #     elif(data_name=='T'or data_name=='F'):
    #         model=copy.deepcopy(model_0)
    #         model.linear=nn.Linear(model.linear.in_features,len(train_dl.dataset.dataset.classes))

    #     else:
    #         model=model_create(model_algo,data_name,len(train_dl.dataset.classes))

        
    #     optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    #     criterion = nn.CrossEntropyLoss() 
    #     model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name, model_algo,root)

    #     torch.cuda.empty_cache() 
    #     return model


    def model_train(model_algo,train_dl,valid_dl,data_name,epochs,model_0=None,root=None):

            if(data_name=="decision"):
                # model=model_create(model_algo,data_name,2)
                model=copy.deepcopy(model_0)
                model.fc= nn.Linear(1280, 2) 

            elif(data_name=='T'or data_name=='F'):
                model=copy.deepcopy(model_0)
                model.fc=nn.Linear(1280, 100) 
            else:
                model=model_create(model_algo,data_name,len(train_dl.dataset.classes))
    
            
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss() 
            model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo,root)

            torch.cuda.empty_cache() 
            return model


   

    model_decision=model_train(model_algo,train_dl,valid_dl,"decision",100,model_0=model_0,root=split_mode)
    # model_decision = torch.load(f'model_{model_algo}_decision_24.pth')

    Fdataset_v=Subset(validset, indexF_v)
    Tdataset_v=Subset(validset, indexT_v)

    Fdl_v=DataLoader(Fdataset_v, shuffle=True, batch_size=batch_size)
    Tdl_v=DataLoader(Tdataset_v, shuffle=True, batch_size=batch_size)

    #訓練模型T、F、decision
    model_T=model_train(model_algo,Tdl,Tdl_v,"T",100,model_0=model_0,root=split_mode)
    # model_T=torch.load(f'model_{model_algo}_T_1.pth')

    #訓練模型T、F、decision
    model_F=model_train(model_algo,Fdl,Fdl_v,"F",100,model_0=model_0,root=split_mode)
    # model_F=torch.load(f'model_{model_algo}_F_1.pth')

    decision_loss,decision_accu,confusion_matrix_decision_train=evaluate_model(model_decision,train_dl,len(trainset),'decision',mode='eval')
    # confusion_matrix_decision_train= confusion_matrix_decision_train.rename(columns={"0":"False","1":"True"}, index={"0":"False","1":"True"})

    decision_valid_loss,decision_valid_accu,confusion_matrix_decision_valid=evaluate_model(model_decision,valid_dl,len(validset),'decision',mode='eval')
    # confusion_matrix_decision_valid= confusion_matrix_decision_valid.rename(columns={0:"False",1:"True"}, index={0:"False",1:"True"})

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
    model_algos=['M_0','M_D',"M_T","M_F","Total_Model",'Total_Model_with_perfect_decision']
    data = {
        'Model': model_algos,
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Valid Loss': valid_losses,
        'Valid Accuracy': valid_accuracies
    }
    df = pd.DataFrame(data)
    df['split_mode'] =str(split_mode)
    #df['split_mode'] =str(split_mode)+str(targetF)
    df['len(D_F):len(D_T) = '] =str("(")+str(len(indexF))+str(" , ")+str(len(indexT))+str(")")

    print(f'split_mode = {split}')
    print(df)

    return df,indexF,indexT,indexF_v,indexT_v

import pickle



#100_f1
i = 10
while i < 100:
    split_mode = ('classf1',i)

    write_header=True
    df,indexF,indexT,indexF_v,indexT_v=training_loop(split_mode)
    df.to_csv('vgg_output_cifar100_classf1.csv', mode='a', header=write_header, index=False)
    write_header = False
    empty_df = pd.DataFrame([[]])
    empty_df.to_csv('vgg_output_cifar100_classf1.csv', mode='a', header=False, index=False)
        
    data = {'indexF': indexF, 'indexT': indexT,'indexF_v': indexF_v, 'indexT_v': indexT_v}
    with open(f'{split_mode[0]}_indices_{i}.pkl', 'wb') as f:
        pickle.dump(data, f)

    i+=10


    print(i)

# i = 0.9

# split_mode = ('softmax',i)
# write_header=True
# df,indexF,indexT,indexF_v,indexT_v=training_loop(split_mode)
# df.to_csv('vgg16_output_cifar100_softmax.csv', mode='a', header=write_header, index=False)
# write_header = False
# empty_df = pd.DataFrame([[]])
# empty_df.to_csv('vgg16_output_cifar100_softmax.csv', mode='a', header=False, index=False)
# data = {'indexF': indexF, 'indexT': indexT,'indexF_v': indexF_v, 'indexT_v': indexT_v}
# with open(f'vgg16_{split_mode[0]}_indices_{i}.pkl', 'wb') as f:
#     pickle.dump(data, f) 

       



#only classaccu targetF line 150
#line 162,165,168
    

    

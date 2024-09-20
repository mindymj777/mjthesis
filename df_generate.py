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
sys.path.append(r'E:/mjpaper/cifar100')
from MJPytorch import *
import os
os.chdir(r'E:/mjpaper/cifar100')



def gerate_decision_df(split_mode):
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

    model_algo="alexnet"
    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainset=CifarDataset(trainset)


    validset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    validset=CifarDataset(validset)

    batch_size =32 # larger numbers lead to CUDA running out of memory
    train_dl = DataLoader(trainset,shuffle=True, batch_size=batch_size)
    valid_dl = DataLoader(validset,shuffle=True, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    # model_0=model_train(model_algo,train_dl,valid_dl,"0",100)
    model_0 = torch.load(r'E:/mjpaper/cifar100/model_alexnet_0_92.pth')
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
    each_f1
    import pickle
    


    if(split_mode[0]=='classf1'):
        targetF=[x  for x in each_f1.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    elif(split_mode[0]=='classaccu'):
        targetF=[x  for x in each_accu.nsmallest(split_mode[1]).index.map(trainset.classess)]
        split=(split_mode[0],targetF)
    else:
        split=split_mode

    # with open(f'E:/mjpaper/cifar100/TandF/T_indices_{split_mode[1]}.pkl', 'rb') as f:
    with open(f'E:/mjpaper/cifar100/TandF/T_indices.pkl', 'rb') as f:
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
    model_decision = torch.load(f'E:/mjpaper/cifar100/TandF/model_alexnet_decision_{split_mode}.pth')
    print("Load model decision OK.")

    #訓練模型T、F、decision
    # model_T=model_train(model_algo,Tdl,Tdl_v,"T",100,model_0=model_0,root=split_mode)
    model_T=torch.load(f'E:/mjpaper/cifar100/TandF/model_alexnet_T_{split_mode}.pth')
    print("Load model_T OK.")

    #訓練模型T、F、decision
    # model_F=model_train(model_algo,Fdl,Fdl_v,"F",100,model_0=model_0,root=split_mode)
    model_F=torch.load(f'E:/mjpaper/cifar100/TandF/model_alexnet_F_{split_mode}.pth')
    print("Load model_F OK.")



    #全部模型裝在一起的表現
    def total_model_evaluate(data_dl,size,model_0,model_T,model_F,model_decision,decision_mode=None,delta1=None,delta2=None):
        total_loss = 0
        accu = 0
        flat_true=[]
        flat_pred=[]

        model_0.eval()
        model_T.eval()
        model_F.eval()
        model_decision.eval()

        with torch.no_grad():
            for data,target,decision_target,idx in data_dl:
                outputs=[]
                data,target=data.to(device),target.to(device)
            

                for d,t in zip(data,target):
                    out_d=model_decision(d.unsqueeze(0))
                    _, pred= torch.max(out_d, dim = 1)
                    softmax_d = torch.max(torch.softmax(out_d, dim=1)).item()
                    
                    out_0=model_0(d.unsqueeze(0))
                    out_F=model_F(d.unsqueeze(0))
                    out_T=model_T(d.unsqueeze(0))

                    softmax_0 = torch.max(torch.softmax(out_0, dim=1)).item()
                    softmax_F = torch.max(torch.softmax(out_F, dim=1)).item()
                    softmax_T = torch.max(torch.softmax(out_T, dim=1)).item()
                
                    if(decision_mode=='model'):
                        if(pred==0):
                            outputs.append(model_F(d.unsqueeze(0)))
                        else:
                            outputs.append(model_T(d.unsqueeze(0)))
                
                    elif(decision_mode=='softmax_0'):

                        if(softmax_0<delta1):
                            outputs.append(out_F)
                    
                        else:
                            outputs.append(out_T)
    
                    elif(decision_mode=='softmax_tf'):
                        if(softmax_F>softmax_T):
                            outputs.append(out_F)
    
                        else:
                            outputs.append(out_T)
    

                    # elif(decision_mode=='model+softmax_d'):
                    #     if(softmax_d<delta1 and abs(softmax_F-softmax_T)>delta2):
                    #         if(softmax_F>softmax_T):
                    #             outputs.append(out_F)

                    #         else:
                    #             outputs.append(out_T)
 
                    #     elif(softmax_d>=delta1 and abs(softmax_F-softmax_T)<=delta2):
                    #         if(pred==0):
                    #             outputs.append(model_F(d.unsqueeze(0)))

                    #         else:
                    #             outputs.append(model_T(d.unsqueeze(0)))
 
                    #     else:
                    #         outputs.append((out_F + out_T + out_0) / 3.0)
                    

                outputs = torch.cat(outputs, dim=0)
                _, y_pred_tag = torch.max(outputs, dim = 1)
                loss = criterion(outputs, target)


                flat_true.extend(target.cpu().numpy())
                flat_pred.extend(y_pred_tag.cpu().numpy())

                total_loss+= loss.item()*data.size(0)
                correct=torch.sum(y_pred_tag == target).item()
                accu += correct      

        total_loss=total_loss/size
        accu=accu/size
        idx2class = {v: k for k, v in data_dl.dataset.classes}
        confusion_matrix_total = pd.DataFrame(confusion_matrix(flat_true, flat_pred)).rename(columns=idx2class, index=idx2class)
        return total_loss,accu,confusion_matrix_total


    decision_loss,decision_accu,confusion_matrix_decision_train=evaluate_model(model_decision,train_dl,len(trainset),'decision',mode='eval')
    decision_valid_loss,decision_valid_accu,confusion_matrix_decision_valid=evaluate_model(model_decision,valid_dl,len(validset),'decision',mode='eval')

    T_loss,T_accu,confusion_matrix_T_train=evaluate_model(model_T,Tdl,len(Tdataset),'T',mode='eval')
    F_loss,F_accu,confusion_matrix_F_train=evaluate_model(model_F,Fdl,len(Fdataset),'F',mode='eval')

    T_valid_loss,T_valid_accu,confusion_matrix_T_valid=evaluate_model(model_T,Tdl_v,len(Tdataset_v),'T',mode='eval')
    F_valid_loss,F_valid_accu,confusion_matrix_F_valid=evaluate_model(model_F,Fdl_v,len(Fdataset_v),'F',mode='eval')

    total_train_loss_d,total_train_accu_d,confusion_matrix_total_train_d=total_model_evaluate_notdecision(train_dl,train_size,model_0,model_T,model_F)
    total_valid_loss_d,total_valid_accu_d,confusion_matrix_total_valid_d=total_model_evaluate_notdecision(valid_dl,valid_size,model_0,model_T,model_F)

    total_train_loss,total_train_accu,confusion_matrix_total_train=total_model_evaluate(train_dl,train_size,model_0,model_T,model_F,model_decision,decision_mode='model')
    total_valid_loss,total_valid_accu,confusion_matrix_total_valid=total_model_evaluate(valid_dl,valid_size,model_0,model_T,model_F,model_decision,decision_mode='model')

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
    # df['split_mode'] =str(split_mode)+str(targetF)
    df['len(Fdata):len(Tdata) = '] =str("(")+str(len(indexF))+str(" , ")+str(len(indexT))+str(")")

    print(f'split_mode = {split}')
    
    #計算model_selection_accu

    def evaluate_md_st(mode,size,data_dl,delta1=None,delta2=None):
        accu = 0
        correct=0
        with torch.no_grad():
            for data,target,decision_target,idx in data_dl:
                torch.cuda.empty_cache()
                outputs=[]
                data,target,decision_target=data.to(device),target.to(device),decision_target.to(device)
                if mode=='decision':         
                    out = model_decision(data)
                    _, y_pred_tag = torch.max(out, dim = 1)
                    correct=torch.sum(y_pred_tag == decision_target).item()
                    accu += correct 

                elif mode=='softmax_tf':
                    for d,dt in zip(data,decision_target):
                        out_F=model_F(d.unsqueeze(0))
                        out_T=model_T(d.unsqueeze(0))

                        softmax_F = torch.max(torch.softmax(out_F, dim=1)).item()
                        softmax_T = torch.max(torch.softmax(out_T, dim=1)).item()

                        if(softmax_F>softmax_T):
                            out=0
                        else:
                            out=1
                        if (out==dt):
                            correct+=1
                    accu = correct
                elif mode=='softmax_0':
                    
                    for d,dt in zip(data,decision_target):
                        out_0=model_0(d.unsqueeze(0))
                        softmax_0 = torch.max(torch.softmax(out_0, dim=1)).item()
                        if(softmax_0<delta1):
                            out=0
                        else:
                            out=1
                        if (out==dt):
                            correct+=1
                    accu = correct

                elif(mode=='model+softmax_d'):
                        for d,dt in zip(data,decision_target):
                            out_F=model_F(d.unsqueeze(0))
                            out_T=model_T(d.unsqueeze(0))
                            out_d=model_decision(d.unsqueeze(0))

                            pred = torch.max(out_d, dim = 1)

                            softmax_F = torch.max(torch.softmax(out_F, dim=1)).item()
                            softmax_T = torch.max(torch.softmax(out_T, dim=1)).item()  
                            softmax_d = torch.max(torch.softmax(out_d, dim=1)).item()    

                            if(softmax_d<delta1 and abs(softmax_F-softmax_T)>delta2):
                                if(softmax_F>softmax_T):
                                    out=0
                                else:
                                    out=1
                            elif(softmax_d>=delta1 and abs(softmax_F-softmax_T)<=delta2):
                                if(pred==0):
                                    out=0
                                else:
                                    out=1
                            else:
                                out=dt #如果兩個softmax和decision選不出來跳過
                            if (out==dt):
                                correct+=1
                        accu = correct
                        
            accu/=size
        return accu


    
    

    def generate_df(mode,delta1=None,delta2=None):
        
        decision_accu=evaluate_md_st(mode=mode,size=train_size,data_dl=train_dl,delta1=delta1,delta2=delta2)
        decision_valid_accu=evaluate_md_st(mode=mode,size=valid_size,data_dl=valid_dl,delta1=delta1,delta2=delta2)

        total_train_loss,total_train_accu,confusion_matrix_total_train_tf=total_model_evaluate(train_dl,train_size,model_0,model_T,model_F,model_decision,decision_mode=mode,delta1=delta1,delta2=delta2)
        total_valid_loss,total_valid_accu,confusion_matrix_total_valid_tf=total_model_evaluate(valid_dl,valid_size,model_0,model_T,model_F,model_decision,decision_mode=mode,delta1=delta1,delta2=delta2)

        train_losses=[train_loss,'NA',T_loss,F_loss,total_train_loss,total_train_loss_d]
        train_accuracies=[train_accu,decision_accu,T_accu,F_accu,total_train_accu,total_train_accu_d]
        valid_losses=[valid_loss,'NA',T_valid_loss,F_valid_loss,total_valid_loss,total_valid_loss_d]
        valid_accuracies=[valid_accu,decision_valid_accu,T_valid_accu,F_valid_accu,total_valid_accu,total_valid_accu_d]

        model_algos=['Model_0',f'{mode}_Decision(detla_d={delta1},detla_tf={delta2})',"Model_T","Model_F","Total_Model",'Total_Model_with_perfect_decision']
        data = {
            'Model': model_algos,
            'Train Loss': train_losses,
            'Train Accuracy': train_accuracies,
            'Valid Loss': valid_losses,
            'Valid Accuracy': valid_accuracies
        }
        df = pd.DataFrame(data)
        # print(f'split_mode = {split}')
        
        return df
    
    df_s0_list = []
    
    # 迴圈遍歷 delta 從 0.1 到 0.9，每次增加 0.1
    for delta in [i/10 for i in range(1, 10)]:
        df_s0 = generate_df(mode='softmax_0', delta1=delta)
        print(f"df_s0_{int(delta*10)} IS OK.")
        df_s0_list.append(df_s0)


    df_tf=generate_df(mode='softmax_tf')
    print("df_tf IS OK.")
    
    return df, df_tf,df_s0_list
    # return df




split_mode = ('TandF')
write_header=True
df, df_tf,df_s0_list=gerate_decision_df(split_mode)
# df=gerate_decision_df(split_mode)

df.to_csv('output_cifar100_TandF_d.csv', mode='a', header=write_header, index=False)
write_header = False
empty_df = pd.DataFrame([[]])
empty_df.to_csv('output_cifar100_TandF_d.csv', mode='a', header=False, index=False)

write_header=True
df_tf.to_csv('output_cifar100_TandF_df_tf.csv', mode='a', header=write_header, index=False)
write_header = False
empty_df = pd.DataFrame([[]])
empty_df.to_csv('output_cifar100_TandF_df_tf.csv', mode='a', header=False, index=False)

for idx, df_s0 in enumerate(df_s0_list, start=1):
    delta_str = f'{idx:02}'  # 將 index 轉換為 01, 02, ..., 09 格式
    write_header = True
    df_s0.to_csv(f'output_cifar100_TandF_df_s0_{delta_str}.csv', mode='a', header=write_header, index=False)
    write_header = False
    empty_df = pd.DataFrame([[]])
    empty_df.to_csv(f'output_cifar100_TandF_df_s0_{delta_str}.csv', mode='a', header=False, index=False)


 

    

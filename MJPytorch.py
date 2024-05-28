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
import copy
import os
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
criterion = nn.CrossEntropyLoss()

#建立dataset class
class CifarDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.images =  dataset        
        self.classess= self.images.class_to_idx
        self.classes = self.images.class_to_idx.items()
        self.flag= [True] * len(self.images)

    def __len__(self):
        return len(self.images)
   

    def __getitem__(self,idx):
        image, label=self.images[idx]

        return image,label,idx
    
    
    def update_flag(self, idx):
        self.flag[idx] = False


#建立dataset class
class ImageDataset(Dataset):
    def __init__(self, root,trans):

        super().__init__()
        self.images = ImageFolder(root=root)        
        self.classess= self.images.class_to_idx
        self.classes = self.images.class_to_idx.items()
        self.flag= [True] * len(self.images)
  
        self.transform = trans


    def __len__(self):
        return len(self.images)
   
    def __getitem__(self,idx):
        image, label=self.images[idx]
        
        image=self.transform(image)

        return image,label,idx
    
    def update_flag(self, idx):
        self.flag[idx] = False


#建立dataset class
class ImagecsvDataset(Dataset):
    def __init__(self, root,csv_path,trans):
        super().__init__()
        self.train_truth=pd.read_csv(csv_path)
        self.path=self.train_truth['filename']
        self.root_dir=root
        self.classess= {label: i for i, label in enumerate (sorted((self.train_truth['label'].unique())))}
        self.classes= {label: i for i, label in enumerate (sorted((self.train_truth['label'].unique())))}.items()
        self.train_truth['label']=self.train_truth['label'].map(self.classess)

        self.flag= [True] * len(self.path)
  
        self.transform = trans

        
        # self.sum=0
    def __len__(self):
        return len(self.path)
   
    def __getitem__(self,idx):
        
        img_path = os.path.join(self.root_dir, self.path[idx]+'.jpeg')
        img = Image.open(img_path)
        
        img=self.transform(img)
        label=torch.tensor(int(self.train_truth['label'][idx]))

        return img,label,idx
    
    def update_flag(self, idx):
        self.flag[idx] = False
    

#建立模型框架
def model_create(model_algo,data_name,class_number):
    model = getattr(models,model_algo)(weights=True)

    if(model_algo=='googlenet' or model_algo=='resnet18' or model_algo=='inception_v3' ):
        num_ftrs=model.fc.in_features
        model.fc= nn.Linear(num_ftrs, class_number)

    # elif(model_algo=='efficientnet_b7' or model_algo=='mobilenet_v2'):
    #     num_ftrs = model.classifier[1].in_features
    #     model.classifier[1]= nn.Linear(num_ftrs, class_number)

    else:
        num_ftrs=model.classifier[6].in_features
        model.classifier[6]= nn.Linear(num_ftrs, class_number)

    return model


#模型表現
def evaluate_model(model,data_dl,size,data_name,mode=None):
        
        model.to(device)
        ######################    
        # validate the model #
        ######################
        torch.cuda.empty_cache()
        total_loss = 0
        accu = 0
        flat_true=[]
        flat_pred=[]
        with torch.no_grad():
            model.eval()   
            for data, target,idx in data_dl:

                    data,target=data.to(device),target.to(device)
                    out= model(data)
                    _, y_pred_tag = torch.max(out, dim = 1)

                    loss = criterion(out, target)
                    # print(target)
                    flat_true.extend(target.cpu().numpy())
                    flat_pred.extend(y_pred_tag.cpu().numpy())

                    total_loss+= loss.item()*data.size(0)
                    correct=torch.sum(y_pred_tag == target).item()
                    accu += correct      

            total_loss=total_loss/size
            accu=accu/size

        a=sorted(set(flat_true))
        if(mode=='eval'):
            if(data_name=='decision'):
                confusion_matrix_valid=pd.DataFrame(confusion_matrix(flat_true, flat_pred),columns=list(a),index=list(a))
            else:
                idxtoclass={v:k for k,v in data_dl.dataset.dataset.classes}
                confusion_matrix_valid= pd.DataFrame(confusion_matrix(flat_true, flat_pred),columns=list(a),index=list(a)).rename(columns=idxtoclass,index=idxtoclass)
            return total_loss,accu,confusion_matrix_valid
        else:
            return total_loss,accu



#訓練模型
def trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo):
    train_accus=[]
    val_accus=[]
    best_val_accu = 0.0
    epochs_without_improvement = 0
    patience=4
    best_model = None
    
    for epoch in range(epochs):
        
        train_loss=0
        model.to(device)
        model.train()
        ###################
        # train the model #
        ###################
        for data, target,idx in train_dl:
                
            optim.zero_grad()
            data,target=data.float().to(device),target.to(device)  #將data、target放到gpu上
            out= model(data)
            # _, y_pred_tag = torch.max(out, dim = 1)  
            loss = criterion(out, target)        
            loss.backward()
            optim.step()
            data,target=data.cpu(),target.cpu()
            # del data,target
            # print(target)
        
        train_loss,train_accu=evaluate_model(model,train_dl,len(train_dl.dataset),data_name)
        print(f"Epoch={epoch},train_loss={train_loss},train_accu={train_accu}")
        train_accus.append(train_accu)
        
        val_loss,val_accu=evaluate_model(model,valid_dl,len(valid_dl.dataset),data_name)
        print(f"Epoch={epoch},valid_loss={val_loss},valid_accu={val_accu}")

        val_accus.append(val_accu)

        # Early stopping
        if val_accu > best_val_accu:
            best_val_accu = val_accu
            epochs_without_improvement = 0
            best_model=copy.deepcopy(model)
            print(f"{epochs_without_improvement}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"{epochs_without_improvement} Early stopping!")
            break
        
        torch.save(best_model, f"model_{model_algo}_{data_name}_{epoch}.pth") 
        torch.cuda.empty_cache()               
    plt.plot(train_accus,'-o')
    plt.plot(val_accus,'-o')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Valid'])
    plt.show()

    torch.save(best_model, f"model_{model_algo}_{data_name}.pth")  
    return best_model

             
#將模型建立框架後並訓練
def model_train(model_algo,train_dl,valid_dl,data_name,epochs,model_0=None):

    if(data_name=="decision"):
        # model=model_create(model_algo,data_name,2)
        model=copy.deepcopy(model_0)
        model.classifier[6]=nn.Linear(model.classifier[6].in_features,2)
    elif(data_name=='T'or data_name=='F'):
        model=copy.deepcopy(model_0)
        model.classifier[6]=nn.Linear(model.classifier[6].in_features,len(train_dl.dataset.dataset.classes))

    else:
        model=model_create(model_algo,data_name,len(train_dl.dataset.dataset.classes))

    if(model_algo=='googlenet' or model_algo=='resnet18' or model_algo=='inception_v3' ):
        model_fc_layer=model.fc
    # elif(model_algo=='efficientnet_b7' or model_algo=='mobilenet_v2'):
    #     model_fc_layer=model.classifier[1]
    else:
        model_fc_layer=model.classifier[6]


    if(data_name=="decision"):
        optim = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss() 
        model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo)
    else:   
        optim = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo)
    torch.cuda.empty_cache() 
    return model




def plot_confusion_matrix(confusion_matrix_train,confusion_matrix_valid,model_name,model_algo):
    fig = plt.figure(figsize=(45, 15))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    sns.heatmap(confusion_matrix_train, annot=True, fmt='',cbar=False,ax=ax1,square=True).set(title=f"{model_algo}_{model_name} train confusion matrix", xlabel="Predicted Label", ylabel="True Label")
    sns.heatmap(confusion_matrix_valid, annot=True, fmt='',ax=ax2,square=True).set(title=f"{model_algo}_{model_name} valid confusion matrix", xlabel="Predicted Label", ylabel="True Label")

def calculate_wmse(epoch,epochs,out,target):
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss=criterion(out,target)
    return epoch/((epochs*(epochs+1))/2)*loss


#將資料切分成true and false
def split_data(model_0,data_dl,split_mode):
    model_0.eval()
    indexF=[]
    indexT=[]
    with torch.no_grad():
        for (data,target,idx) in data_dl:
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
                        data_dl.dataset.dataset.update_flag(idx)
                    else:
                        indexT.append(idx.cpu().numpy().item())  
                elif(split_mode=='TandF'):
                    
                    if(t!=pred):
                        indexF.append(idx.cpu().numpy().item())  
                        data_dl.dataset.dataset.update_flag(idx)
                    else:
                        indexT.append(idx.cpu().numpy().item())
                     
                   
                elif(split_mode[0]=='softmax'):
                 
                    if(softmax.max()<split_mode[1]):
                        indexF.append(idx.cpu().numpy().item())  
                        data_dl.dataset.dataset.update_flag(idx)
                    else:
                        indexT.append(idx.cpu().numpy().item())
                        # if(softmax.max()<split_mode[2]):
                        #     indexF.append(idx.cpu().numpy().item())

                elif(split_mode[0]=='classaccu'):
                    if(t in split_mode[1]):
                        indexF.append(idx.cpu().numpy().item())  
                        data_dl.dataset.dataset.update_flag(idx)
                        # if(t in [6,4,7,2]):
                        #     indexT.append(idx.cpu().numpy().item()) 
                            
                    else:
                        indexT.append(idx.cpu().numpy().item()) 
                        # if(t in [6,4,7,2]):
                        #     indexF.append(idx.cpu().numpy().item()) 
                    
                     # elif(split_mode[0]=='wmse'):
                #     if(data_dl.dataset.dataset.loss[idx]>split_mode[1]):
                #         indexF.append(idx.cpu().numpy().item())  
                #         data_dl.dataset.dataset.update_flag(idx)
                #     else:
                #         indexT.append(idx.cpu().numpy().item())  
               
                            
        torch.cuda.empty_cache() 
    return indexF,indexT
                
               
#decision set
def decision_split(data_dl,model_0):
    flat_data=[]
    flat_true=[]
    with torch.no_grad():

        model_0.eval()   

        for data, target,idx in data_dl:

            data,target=data.to(device),target.to(device)
            out= model_0(data)
            softmax = torch.softmax(out, dim=1)
            _, y_pred_tag = torch.max(out, dim = 1) 

            for i,d,t in zip(idx,data,target):
                flat_true.append(int(data_dl.dataset.dataset.flag[i]))
                flat_data.append(d.cpu().numpy()) 
            
    return flat_data,flat_true

#建立dataset class
class TandFDataset(Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.x = x        
        self.y = y
        self.classes={label: i for i, label in enumerate (sorted(list(set(self.y))))}
    
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):

        image=self.x[idx]
  
        label=torch.tensor(self.classes[self.y[idx]])

        return image,label,idx
    def get_labels(self):
        return [self.classes[label] for label in self.y]
    
    


#建立dataset class
class DecisionDataset(Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.x = x        
        self.y = y
        self.classes={0,1}
    
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):

        image=self.x[idx]
  
        label=self.y[idx]
        return image,label,idx
    def get_labels(self):
        return self.y
    


#全部模型裝在一起的表現
def total_model_evaluate(data_dl,size,model_0,model_T,model_F,model_decision):
    total_loss = 0
    accu = 0
    flat_true=[]
    flat_pred=[]
    sum_F=0
    sum_T=0
    with torch.no_grad():
        for data,target,idx in data_dl:
            outputs=[]
            data,target=data.to(device),target.to(device)
            out_growth = model_0(data)

            softmax=torch.softmax(out_growth, dim=1)

            out=model_decision(data)
            _, y_pred_tag = torch.max(out, dim = 1)


            for pred,d,t in zip(y_pred_tag,data,target):

                if(pred==0):
                    outputs.append(model_F(d.unsqueeze(0)))
                    sum_F+=1
                else:
                    outputs.append(model_T(d.unsqueeze(0)))
                    sum_T+=1

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
    idx2class = {v: k for k, v in data_dl.dataset.dataset.classes}
    confusion_matrix_total = pd.DataFrame(confusion_matrix(flat_true, flat_pred)).rename(columns=idx2class, index=idx2class)
    return total_loss,accu,confusion_matrix_total,sum_F,sum_T


#全部模型裝在一起的表現
def total_model_evaluate_notdecision(data_dl,size,model_0,model_T,model_F):
    total_loss = 0
    accu = 0
    flat_true=[]
    flat_pred=[]
    sum_F=0

    with torch.no_grad():
        for data,target,idx in data_dl:
            outputs=[]
            data,target=data.to(device),target.to(device)
            out_growth = model_0(data)
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss_0 = criterion(out_growth , target)
     
            for l,d,t,idx in zip(loss_0,data,target,idx):
                
                if(data_dl.dataset.dataset.flag[idx]==False):
                    outputs.append(model_F(d.unsqueeze(0)))
                    sum_F+=1
                else:
                    outputs.append(model_T(d.unsqueeze(0)))
                    
            criterion = nn.CrossEntropyLoss()
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
    idx2class = {v: k for k, v in data_dl.dataset.dataset.classes}
    confusion_matrix_total = pd.DataFrame(confusion_matrix(flat_true, flat_pred)).rename(columns=idx2class, index=idx2class)
    return total_loss,accu,confusion_matrix_total,sum_F



# def calculate_lcb(data_dl,model):

#     softmax_max_list = []

#     # 计算每张图片的 softmax 最大值并添加到列表中
#     with torch.no_grad():
#         for images, labels in data_dl:
#             images = images.to(device)
#             outputs = model(images)
#             softmax_outputs = nn.softmax(outputs, dim=1)
#             max_values, _ = torch.max(softmax_outputs, dim=1)
#             softmax_max_list.extend(max_values.cpu().numpy())

#     # 计算平均值
#     avg = sum(softmax_max_list) / len(softmax_max_list)
#     std = torch.tensor(softmax_max_list).std().item()

#     lcb=avg-std

#     return lcb

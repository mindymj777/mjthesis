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
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
criterion = nn.CrossEntropyLoss()

# CIFAR100 Dataset 
class CifarDataset(Dataset):
    def __init__(self, dataset,transform=None):
        super().__init__()
        self.images =  dataset        
        self.classess= self.images.class_to_idx
        self.classes = self.images.class_to_idx.items()
        self.flag= [True] * len(self.images)
    
        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        decision_label = torch.tensor(int(self.flag[idx]))
        return image, label, decision_label, idx
    
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
        decision_label=torch.tensor(int(self.flag[idx]))

        return image,label,decision_label,idx
    
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
        
        img_path = os.path.join(self.root_dir, self.path[idx])
        img = Image.open(img_path)
        
        img=self.transform(img)
        label=torch.tensor(int(self.train_truth['label'][idx]))
        decision_label=torch.tensor(int(self.flag[idx]))

        return img,label,decision_label,idx
    
    def update_flag(self, idx):
        self.flag[idx] = False



class Alexnet(nn.Module):
    def __init__(self,class_number):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, class_number)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
      
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
  
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=100, width_mult=0.5, dropout_prob=0.3):
#         super(MobileNetV2, self).__init__()
       
#         self.mobilenet_v2 = models.mobilenet_v2(pretrained=True, width_mult=width_mult)
        

#         self.mobilenet_v2.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_prob),
#             nn.Linear(self.mobilenet_v2.classifier[1].in_features, num_classes)
#         )
        
#     def forward(self, x):
#         return self.mobilenet_v2(x)


# class EfficientNetB0(nn.Module):
#     def __init__(self, n_classes, dropout_rate=0.7):
#         super(EfficientNetB0, self).__init__()
    
#         efficientnet_b0 = models.efficientnet_b0(pretrained=True)

#         self.base_model = nn.Sequential(*list(efficientnet_b0.children())[:-1])
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(1280, n_classes) 

#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x  


class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 -> 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 定義 DenseNet Block
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.avg_pool(out)
        return out

class DenseNetBC(nn.Module):
    def __init__(self, num_classes=100, growth_rate=12, reduction=0.5, num_layers=(6, 6, 6)):
        super(DenseNetBC, self).__init__()
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        self.block1 = DenseBlock(num_layers[0], num_channels, growth_rate)
        num_channels += num_layers[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = TransitionLayer(num_channels, out_channels)
        num_channels = out_channels
        
        self.block2 = DenseBlock(num_layers[1], num_channels, growth_rate)
        num_channels += num_layers[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = TransitionLayer(num_channels, out_channels)
        num_channels = out_channels
        
        self.block3 = DenseBlock(num_layers[2], num_channels, growth_rate)
        num_channels += num_layers[2] * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


#建立模型框架
def model_create(model_algo,data_name,class_number):
    if model_algo == 'alexnet':
        model = Alexnet()
    # elif model_algo=='mobilenet_v2':
    #     model=MobileNetV2()
    # elif model_algo=='efficientnet_b0':
    #     model = EfficientNetB0(n_classes=class_number)
    # elif model_algo=='squeezenet':
    #     model=SqueezeNet()
    elif model_algo=='resnet20':
        model=ResNet20()
    # elif model_algo=='ShuffleNetV2':
    #     model=ShuffleNetV2()
    elif model=='vgg16':
        model=VGG16()
        
    elif model_algo=='DenseNetBC':
        model=DenseNetBC()
    else:
        model = getattr(models,model_algo)(weights=True)

        if(model_algo=='googlenet' or model_algo=='resnet18' or model_algo=='inception_v3' ):
            num_ftrs=model.fc.in_features
            model.fc= nn.Linear(num_ftrs, class_number)

        elif(model_algo=='efficientnet_b7' or model_algo=='mobilenet_v2'):
            num_ftrs = model.classifier[1].in_features
            model.classifier[1]= nn.Linear(num_ftrs, class_number)


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
            for data, target,decision_target,idx in data_dl:

                    data,target,decision_target=data.to(device),target.to(device),decision_target.to(device)
                    out= model(data)
                    _, y_pred_tag = torch.max(out, dim = 1)
                    if(data_name=='decision'):
                        loss = criterion(out, decision_target)
                        correct=torch.sum(y_pred_tag == decision_target).item()
                        flat_true.extend(decision_target.cpu().numpy())
                    else:
                        loss = criterion(out, target)
                        correct=torch.sum(y_pred_tag == target).item()
                        flat_true.extend(target.cpu().numpy())
          
                    
                    flat_pred.extend(y_pred_tag.cpu().numpy())

                    total_loss+= loss.item()*data.size(0)

                    
                    accu += correct      

            total_loss/=size
            accu/=size

        all_classes = sorted(set(flat_true) | set(flat_pred))
        if(mode=='eval'):
            if(data_name=='decision'):
                confusion_matrix_valid=pd.DataFrame(confusion_matrix(flat_true, flat_pred, labels=all_classes),columns=list(all_classes),index=list(all_classes))
            elif(data_name=='T'or data_name=='F'):
                idxtoclass={v:k for k,v in data_dl.dataset.dataset.classes}
                confusion_matrix_valid=pd.DataFrame(confusion_matrix(flat_true, flat_pred, labels=all_classes),columns=list(all_classes),index=list(all_classes)).rename(columns=idxtoclass,index=idxtoclass)
            else:
                idxtoclass={v:k for k,v in data_dl.dataset.classes}
                confusion_matrix_valid= pd.DataFrame(confusion_matrix(flat_true, flat_pred, labels=all_classes),columns=list(all_classes),index=list(all_classes)).rename(columns=idxtoclass,index=idxtoclass)
            return total_loss,accu,confusion_matrix_valid
        else:
            return total_loss,100.*accu



#訓練模型
def trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo,root=None):
    train_accus=[]
    val_accus=[]
    best_val_accu = 0.0
    epochs_without_improvement = 0
    patience=5
    best_model = None

    cnt_for_lr_reduce = 0
    
    for epoch in range(epochs):
        
        train_loss=0
        model.to(device)
        model.train()
        ###################
        # train the model #
        ###################
        for data, target,decision_target,idx in train_dl:
            
            optim.zero_grad()
            data,target,decision_target=data.float().to(device),target.to(device),decision_target.to(device)  #將data、target放到gpu上                

            out= model(data)
            # _, y_pred_tag = torch.max(out, dim = 1)  
            if(data_name=='decision'):
                
                loss = criterion(out, decision_target)     
            else:
                loss = criterion(out, target)      
            
            loss.backward()
            optim.step()
            data,target,decision_target=data.cpu(),target.cpu(),decision_target.cpu()

        train_loss,train_accu=evaluate_model(model,train_dl,len(train_dl.dataset),data_name)
        print(f"Epoch={epoch},train_loss={train_loss},train_accu={train_accu}%")
        train_accus.append(train_accu)
        
        val_loss,val_accu=evaluate_model(model,valid_dl,len(valid_dl.dataset),data_name)
        print(f"Epoch={epoch},valid_loss={val_loss},valid_accu={val_accu}%")

        val_accus.append(val_accu)

        # Early stopping
        if val_accu > best_val_accu:
            epochs_without_improvement = 0
            best_val_accu = val_accu
            cnt_for_lr_reduce=0
            best_model=copy.deepcopy(model)
            print(f"{epochs_without_improvement}")

            # torch.save(best_model, f"model_{model_algo}_{data_name}.pth")
        else:
            epochs_without_improvement += 1
            cnt_for_lr_reduce += 1
            print(f'EarlyStopping counter: {epochs_without_improvement} out of {patience}')

        if cnt_for_lr_reduce >=2:
            for p in optim.param_groups:
                p['lr'] *= 0.8
            cnt_for_lr_reduce = 0
            print(f"Learning rate reduced by 10%{p['lr']}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping!")
            break
        
         
        torch.cuda.empty_cache()               
    # plt.plot(train_accus,'-o')
    # plt.plot(val_accus,'-o')
    # plt.xlabel('epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(['Train','Valid'])
    # plt.show()

    torch.save(best_model, f"model_{model_algo}_{data_name}_{root}.pth")  
    return best_model

             
def model_train(model_algo,train_dl,valid_dl,data_name,epochs,model_0=None,root=None):

        if(data_name=="decision"):
            model=copy.deepcopy(model_0)
            model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)

        elif(data_name=='T'or data_name=='F'):
            model=copy.deepcopy(model_0)
            model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(train_dl.dataset.dataset.classes))

        else:
            model=model_create(model_algo,data_name,len(train_dl.dataset.classes))
 
        
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss() 
        model=trainer(epochs,model,criterion,optim,train_dl,valid_dl,data_name,model_algo,root)

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
                        # if(t in split_mode[2]):
                        #     indexT.append(idx.cpu().numpy().item())    
                    else:
                        indexT.append(idx.cpu().numpy().item())
                        # if(t in split_mode[2]):
                        #     indexF.append(idx.cpu().numpy().item())           
                            
        torch.cuda.empty_cache() 
    return indexF,indexT

 
#全部模型裝在一起的表現
def total_model_evaluate(data_dl,size,model_0,model_T,model_F,model_decision,decision_mode=None,delta1=None,delta2=None):
    total_loss = 0
    accu = 0
    flat_true=[]
    flat_pred=[]
    sum_F=0
    sum_T=0
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
                        sum_F+=1
                    else:
                        outputs.append(model_T(d.unsqueeze(0)))
                        sum_T+=1
                elif(decision_mode=='softmax_0'):
                   
                    if(softmax_0<delta1):
                        outputs.append(out_F)
                        sum_F+=1
                    else:
                        outputs.append(out_T)
                        sum_T+=1

                elif(decision_mode=='softmax_tf'):
                    if(softmax_F>softmax_T):
                        outputs.append(out_F)
                        sum_F+=1
                    else:
                        outputs.append(out_T)
                        sum_T+=1

                elif(decision_mode=='model+softmax_d'):
                    if(softmax_d<delta1 and abs(softmax_F-softmax_T)>delta2):
                        if(softmax_F>softmax_T):
                            outputs.append(out_F)
                            sum_F+=1
                        else:
                            outputs.append(out_T)
                            sum_T+=1
                    elif(softmax_d>=delta1 and abs(softmax_F-softmax_T)<=delta2):
                        if(pred==0):
                            outputs.append(model_F(d.unsqueeze(0)))
                            sum_F+=1
                        else:
                            outputs.append(model_T(d.unsqueeze(0)))
                            sum_T+=1
                    else:
                        outputs.append((out_F + out_T + out_0) / 3.0)
                   

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
    return total_loss,accu,confusion_matrix_total,sum_F,sum_T



#全部模型裝在一起的表現
def total_model_evaluate_notdecision(data_dl,size,model_0,model_T,model_F):
    total_loss = 0
    accu = 0
    flat_true=[]
    flat_pred=[]
    sum_F=0

    with torch.no_grad():
        for data,target,decision_target,idx in data_dl:
            outputs=[]
            data,target,decision_target=data.to(device),target.to(device),decision_target.to(device)
            
     
            for d,t,idx in zip(data,target,idx):
                
                if(data_dl.dataset.flag[idx]==False):
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
    idx2class = {v: k for k, v in data_dl.dataset.classes}
    confusion_matrix_total = pd.DataFrame(confusion_matrix(flat_true, flat_pred)).rename(columns=idx2class, index=idx2class)
    return total_loss,accu,confusion_matrix_total,sum_F


def show_model_evaluate(data_dl,size,model_0,model_T,model_F,model_decision):
    total_loss = 0
    accu = 0
    flat_true=[]
    flat_pred=[]
    results = []
    
    with torch.no_grad():
        for data,target,_,idx in data_dl:
   
            data,target=data.to(device),target.to(device) 
            
            for d,t,idx in zip(data,target,idx):
                
                output_F=model_F(d.unsqueeze(0))
                output_T=model_T(d.unsqueeze(0))
                output_0=model_0(d.unsqueeze(0))
                output_decision=model_decision(d.unsqueeze(0))
                # print(output_F)
                pred_F=torch.max(output_F, dim = 1).indices.item()
                pred_T=torch.max(output_T, dim = 1).indices.item()
                pred_0=torch.max(output_0, dim = 1).indices.item()
                pred_decision=torch.max(output_decision, dim = 1).indices.item()

                softmax_F=torch.max(torch.softmax(output_F, dim=1)).item()
                softmax_T=torch.max(torch.softmax(output_T, dim=1)).item()
                softmax_0=torch.max(torch.softmax(output_0, dim=1)).item()
                softmax_decision=torch.max(torch.softmax(output_decision, dim=1)).item()
                
                # Record results in a dictionary
                result_dict = {
                    'Index': idx.item(),
                    'Target': t.item(),
                    'Pred_F': pred_F,
                    'Pred_T': pred_T,
                    'Pred_0': pred_0,

                    'Softmax_F': softmax_F,
                    'Softmax_T': softmax_T,
                    'Softmax_0': softmax_0,
                    'Softmax_decision': softmax_decision,
                    
                    'Match_F': pred_F == t.item(),
                    'Match_T': pred_T == t.item(),
                    'Match_0': pred_0 == t.item(),
                    'Decision':bool(pred_decision),
                    'flag':data_dl.dataset.flag[idx],
                    'Match_d': bool(pred_decision) == data_dl.dataset.flag[idx]
                }

                # Append result dictionary to the results list
                results.append(result_dict)

    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results)
    return  df_results
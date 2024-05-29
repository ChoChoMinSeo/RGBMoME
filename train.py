import torch
import torch.nn as nn
from torchvision.transforms import v2
import pandas as pd
from tqdm import tqdm

from config import fix_seed, config, save_paths
from dataloader import train_loader, valid_loader
from utils.count_parameter import model_analysis
from model.rgb_mome import RGB_MoME

fix_seed(config['seed'])

model = RGB_MoME(
    num_channels = 3,
    img_size = 224,
    patch_size = 16,
    emb_dim = 768,
    num_head = 12,
    sep_ffn_dim = 1024,
    comb_ffn_dim = 3072,
    sep_channel_depth = 6,
    combine_channel_depth = 18,
    num_classes = 100,
    ffn_dropout = 0.1,
    dropout = 0.1,
    activation_fn = 'gelu',
    task_type = 'classification',
    include_channel_embedding = True,
    relative_pos=True,
    multiway_patch_embedding=False,
    auxilary_output= config['aux_output']
)

# model.load_state_dict(torch.load('PATH'))

model_analysis(model)

cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def cal_acc(output, target):
    pred= torch.argmax(output,dim=1)
    target = torch.argmax(target, dim=1)
    acc = torch.sum(pred == target)
    return acc
def cal_acc2(output, target):
    pred= torch.argmax(output,dim=1)
    acc = torch.sum(pred == target)
    return acc

def validation(config, model, criterion, valid_loader):
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images = images.to(config['device'],dtype = torch.float32)
            labels = labels.to(config['device'],dtype = torch.int64)
            outputs = model(images)
            acc = cal_acc2(outputs, labels)
            loss = criterion(outputs, labels)
            
            val_loss+=loss.item()
            val_acc+=acc.item()

    return val_loss/len(valid_loader), val_acc/10000

def train(config, model, train_loader, valid_loader):
    model = model.to(config['device'])
    es_count = 0
    best_epoch = 0
    max_val_acc = 0
    best_model = None
    train_df = pd.DataFrame({'train_loss':[],'train_acc':[],'valid_loss':[],'valid_acc':[]})
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr = config['lr'], betas=(0.9,0.999), eps=1e-8,weight_decay = 0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=1, verbose=True)
    
    print('***TRAINING START***')
    
    for epoch in range(config['epochs']):
        model.train()

        epoch_loss = 0
        epoch_acc = 0
        for images, labels in tqdm(train_loader):
            images = images.to(config['device'],dtype = torch.float32)
            labels = labels.to(config['device'],dtype = torch.int64)
            
            images, labels = cutmix_or_mixup(images, labels)
        
            optimizer.zero_grad()
            if config['aux_output']:
                outputs,aux_outputs = model(images)
                aux_loss_r = criterion(aux_outputs[:,:100],labels)
                aux_loss_g = criterion(aux_outputs[:,100:200],labels)
                aux_loss_b = criterion(aux_outputs[:,200:],labels)
                loss = criterion(outputs, labels)/2
                loss += (0.299*aux_loss_r+0.587*aux_loss_g+0.114*aux_loss_b)/2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            acc = cal_acc(outputs, labels)
            epoch_loss+=loss.item()
            epoch_acc+=acc.item()
            
            
        val_loss, val_acc = validation(config, model, criterion, valid_loader)
        train_df.loc[epoch] = [epoch_loss/len(train_loader), epoch_acc/50000, val_loss, val_acc]
        es_count += 1
        if max_val_acc < val_acc:
            es_count = 0
            max_val_acc = val_acc
            best_model = model
            state_dict = model.state_dict()
            best_epoch = epoch+1
            print(f"Epoch [{best_epoch}] New Valid Accuracy!")
            print("..save current best model..")
            model_name = f'epoch {epoch+1}_current_best_model.pt'
            torch.save(state_dict, save_paths['model']+'/'+model_name)
        
        scheduler.step()
        train_df.to_csv('./logs/log.csv')
        if es_count == config['early_stopping']:
            print(f"Early Stopping Count에 도달했습니다!")
            print(f"Epoch {epoch+1}, Best Epoch: {best_epoch}, ES Count: {es_count}\nTrain CE Loss: {(epoch_loss/len(train_loader)):6f}, Train Accuracy: {epoch_acc/50000:6f}, Valid CE Loss: {val_loss:6f}, Valid Accuracy: {val_acc:6f}")
            print("***TRAINING DONE***")
            return best_model
        print(f"Epoch {epoch+1}, Best Epoch: {best_epoch}, ES Count: {es_count}\nTrain CE Loss: {(epoch_loss/len(train_loader)):6f}, Train Accuracy: {epoch_acc/50000:6f}, Valid CE Loss: {val_loss:6f}, Valid Accuracy: {val_acc:6f}")
        print("------------------------------------------------------------------------------------")
    print(f"Early Stopping Count에 도달하지 않았습니다! \nEarly Stopping Count: {config['early_stopping']} Best Epoch: {best_epoch}")
    print("***TRAINING DONE***")
    return best_model

best_model = train(config,model,train_loader,valid_loader)

model_name = 'last_model.pt'
torch.save(best_model.state_dict(), save_paths['model']+'/'+model_name)
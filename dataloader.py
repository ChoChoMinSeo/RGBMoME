from torch.utils.data import Dataset, DataLoader
from config import data_paths, fix_seed, train_csv, test_csv,config
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from sklearn.model_selection import train_test_split

class Cifar_Dataset(Dataset):
    def __init__(self, csv, transform_f=None,infer = False):
        self.csv = csv
        self.transform_f = transform_f
        self.infer = infer
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self,idx):
        img_path = self.csv.iloc[idx,0]
        label = self.csv.iloc[idx,1]
        if self.infer:
            image = Image.open(data_paths['test_img']+img_path)
        else:
            image = Image.open(data_paths['train_img']+img_path)
        
        if self.transform_f:
            image = self.transform_f(image)
        return image, label

base_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(0.1),
        transforms.AutoAugment(policy = torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        # transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.1,contrast = 0.1 ,saturation =0.1 ),
        # transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
        
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ]),
}

fix_seed(config['seed'])
# train_csv_, valid_csv_ = train_test_split(train_csv,test_size=0.1, random_state=config['seed'],stratify=train_csv['label'])
# train_dataset = Cifar_Dataset(train_csv_,transform_f=base_transforms['train'],infer=False)
# valid_dataset = Cifar_Dataset(valid_csv_,transform_f=base_transforms['valid'],infer=False)
# test_dataset = Cifar_Dataset(test_csv,transform_f=base_transforms['valid'],infer=True)


train_dataset = torchvision.datasets.CIFAR100("./cifar100",
                                         train=True,
                                         download=True,
                                         transform=base_transforms['train'])
valid_dataset = torchvision.datasets.CIFAR100("./cifar100",
                                         train=False,
                                         download=True,
                                         transform=base_transforms['valid'])


train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=True, pin_memory = True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],shuffle=False, pin_memory = True)
# test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],shuffle=False,num_workers = 8, pin_memory = True)
# print(f'train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}')
print(f'train size: {len(train_dataset)}, valid size: {len(valid_dataset)}')

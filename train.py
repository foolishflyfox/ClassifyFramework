from config import *
import torchvision as tv
from torchclassify import models
from torchclassify import utils
import torch.backends.cudnn as cudnn
import torchclassify
import torch
import os
import os.path as osp

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

device = torch.device('cpu')

if model_cfg['cuda'] and torch.cuda.is_available():
    device = torch.device('cuda:0')
    cudnn.benchmark = True

model, size = models.initialize_model(model_cfg['model_name'], model_cfg['classify_layers'], 
                                model_cfg['pretrained'])
if model_cfg['load_path'] is not None and osp.isfile(model_cfg['load_path']):
    print(f"Loading model from {model_cfg['load_path']}...")
    model.load_state_dict(torch.load(model_cfg['load_path']))
    for param in model.parameters():
        param.requires_grad_(True)
    print(f"Finish loading model!")

model.to(device)

if train_cfg['train_mode']==TrainMode.FEATURE_EXTRACT:
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.custom_layers.parameters():
        param.requires_grad_(True)

train_dataset, val_dataset = None, None

train_transform = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(imagenet_mean, imagenet_std),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Resize((size, size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(imagenet_mean, imagenet_std),
])

print("Loading dataset...")

if 0 < train_cfg['train_rate'] < 1:
    train_dataset, val_dataset = torchclassify.splited_image_dataset(train_cfg['dir'],
                                    train_transform, train_cfg['train_rate'],
                                    val_transform, train_cfg['shuffle'])
else:
    train_dataset = torchclassify.splited_image_dataset(train_cfg['dir'], train_transform)

if 'dir' in val_cfg and val_cfg['dir'] is not None:
    val_dataset = torchclassify.splited_image_dataset(val_cfg['dir'], val_transform)

print(f"Train dataset\n\tpath:{train_dataset.root}\n\tsize:{len(train_dataset)}")

if val_dataset:
    print(f"Val dataset\n\tpath:{val_dataset.root}\n\tsize:{len(val_dataset)}")

train_dataloader, val_dataloader = None, None
train_dataloader = torch.utils.data.DataLoader(train_dataset, train_cfg['batch_size'], True)

if val_dataset:
    val_dataloader = torch.utils.data.DataLoader(val_dataset, val_cfg['batch_size'], False)

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

if train_cfg['train_mode']==TrainMode.FINE_TUNE:
    params_to_update = model.parameters()
else:
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)

optimizer = torch.optim.RMSprop(params_to_update, lr=train_cfg['lr'])

model, hist = utils.train_model(model, train_dataloader, val_dataloader, criterion, optimizer,
                                num_epochs=train_cfg['num_epochs'], device=device)

if not osp.isdir(train_cfg['train_data_dir']):
    os.makedirs(train_cfg['train_data_dir'])

data_dir = osp.join(train_cfg['train_data_dir'], utils.get_timestamp())
os.makedirs(data_dir)

if train_cfg['save_best']:
    model_save_path = osp.join(data_dir, 'best_model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Save best model to path: {osp.abspath(osp.expanduser(model_save_path))}")


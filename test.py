from config import test_cfg, imagenet_mean, imagenet_std
from torchclassify import utils
from torchclassify import models
from torchvision import transforms
import torch
from torch.backends import cudnn
import os.path as osp
import os
import json
from tqdm import tqdm
import numpy as np


def load_model_test():
    device = torch.device('cpu')

    with open(osp.join(test_cfg['model_dir'], 'all_cfg.json')) as f:
        all_cfg = json.load(f)

    model_cfg = all_cfg['model_cfg']

    if model_cfg['cuda'] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        cudnn.benchmark = True

    model, size = models.initialize_model(model_cfg['model_name'], model_cfg['classify_layers'], False)
    model_path = osp.join(test_cfg['model_dir'], 'best_model.pt')

    if osp.isfile(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        for param in model.parameters():
            param.requires_grad_(False)
        print(f"Finish loading model!")

    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_dataset = utils.TestImageFolder(test_cfg['dir'], test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=test_cfg['batch_size'],
                                                num_workers=test_cfg['num_works'])

    pbar = tqdm(test_dataloader, desc="Test progress")

    test_output = []
    test_imgs_path = []

    for inputs, paths in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_output.append(outputs.to('cpu').numpy())
        test_imgs_path.extend(paths)
    test_output = np.vstack(test_output)

    pbar.close()
    return test_output, test_imgs_path
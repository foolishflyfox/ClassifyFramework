import time
import torch
import copy
from tqdm import tqdm

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, 
            num_epochs, device):
    is_inception = model.name.startswith('inception')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []

    for epoch in range(num_epochs):
        print('-'*16, f"\nEpoch [{epoch+1:3d}/{num_epochs:3d}]")

        train_loss = 0.0
        train_corrects = 0
        model.train()
        with torch.enable_grad():
            pbar = tqdm(train_dataloader, desc="Training  " ,leave=False)
            # all_steps, cur_steps = len(train_dataloader), 0
            for inputs, labels in pbar:
                # cur_steps += 1
                # pbar.set_description(f'training   [{cur_steps}/{all_steps}]')
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                if is_inception:
                    outputs, aux_outputs = model(inputs)
                    step_loss = criterion(outputs, labels)+ 0.4 * criterion(aux_outputs, labels)
                else:
                    outputs = model(inputs)
                    step_loss = criterion(outputs, labels)
                step_loss.backward()
                optimizer.step()
                train_loss += step_loss * inputs.size(0)
                _, prediction = outputs.max(dim=1)
                train_corrects += (prediction==labels).sum()
            pbar.close()
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects.double() / len(train_dataloader.dataset)
        print(f'\ttrain_acc:{train_acc:.6f},\ttrain loss:{train_loss:.6f}')
        
        if val_dataloader is None: continue

        val_corrects = 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Validation", leave=False)
            # all_steps, cur_steps = len(val_dataloader), 0
            for inputs, labels in pbar:
                # cur_steps += 1
                # pbar.set_description(f"validation [{cur_steps}/{all_steps}]")
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, prediction = outputs.max(dim=1)
                val_corrects += (prediction==labels).sum()
            pbar.close()
        val_acc = val_corrects.double() / len(val_dataloader.dataset)
        print(f"\t  val_acc:{val_acc:.6f}")
        val_acc_history.append(val_acc.item())
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print("="*16)
    time_escaped = time.time() - since
    print(f"Training complete in {time_escaped//60:.0f}m {time_escaped%60:.0f}s")
    print(f"Best val Acc: {best_acc:.6f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


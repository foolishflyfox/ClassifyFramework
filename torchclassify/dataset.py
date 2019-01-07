from torchvision.datasets import ImageFolder
import copy
import random

def splited_image_dataset(root, train_transform=None, train_rate=1.0, val_transform=None,
                            shuffle=True, target_transform=None):
    """ Generate two dataset for training and validation
    Args:
        shuffle: shuffle images before split or not        
    """
    kwargs = {
        'root': root,
        'transform': train_transform,
        'target_transform': target_transform,
    }

    if train_rate<=0.0 or train_rate>=1.0:
        return ImageFolder(**kwargs)
    else:
        train_dataset = ImageFolder(**kwargs)
        val_dataset = copy.deepcopy(train_dataset)
        all_samples = train_dataset.samples
        all_size = len(train_dataset)
        train_size = int(all_size * train_rate)
        if shuffle:
            random.shuffle(all_samples)

        train_dataset.imgs = train_dataset.samples = all_samples[:train_size]
        val_dataset.transform = val_transform
        val_dataset.imgs = val_dataset.samples = all_samples[train_size:]
        
        return train_dataset, val_dataset






import torch.utils.data as data
from PIL import Image
import os

# refer to https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None, extensions=IMG_EXTENSIONS):
        self.root = root
        self.extensions = extensions
        self.imgs = []

        for tp_root, _, fnames in os.walk(root):
            for fname in fnames:
                if has_file_allowed_extension(fname, extensions):
                    self.imgs.append(os.path.join(tp_root, fname))
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.imgs)


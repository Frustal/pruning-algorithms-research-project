import ssl
from pathlib import Path
from torchvision.datasets import Flowers102
from torchvision import transforms
from torch.utils.data import DataLoader

ssl._create_default_https_context = ssl._create_unverified_context

def get_dataloaders(batch_size, num_workers=0):
    root = Path("data")
    root.mkdir(exist_ok=True) 

    # transforms
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    eval_t = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])

    train_set = Flowers102(root=root, split="train", transform=train_t, download=True)
    val_set   = Flowers102(root=root, split="val",   transform=eval_t,  download=True)
    test_set  = Flowers102(root=root, split="test",  transform=eval_t,  download=True)

    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val':   DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test':  DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return loaders
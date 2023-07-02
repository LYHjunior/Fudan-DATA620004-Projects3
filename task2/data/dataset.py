from torchvision import datasets, transforms
from data.argument_type import Cutout

def load_cifar_dataset(cfg):
    # Image Preprocessing
    train_transform = transforms.Compose([])
    if cfg.DATA.augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    test_transform = transforms.Compose([
        transforms.ToTensor()])

    if cfg.DATA.cutout:
        train_transform.transforms.append(Cutout(n_holes=cfg.CUTOUT.n_holes, length=cfg.CUTOUT.length))
    
    if cfg.DATA.name == 'cifar-10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root=cfg.DATA.root,
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR10(root=cfg.DATA.root,
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif cfg.DATA.name == 'cifar-100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root=cfg.DATA.root,
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR100(root=cfg.DATA.root,
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    return train_dataset,test_dataset,num_classes

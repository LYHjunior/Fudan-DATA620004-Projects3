# config.py
import os
from torchvision import transforms

use_gpu=True
gpu_name=2
#model_stage1_best.pth
#model.pth
pre_model=os.path.join('pth','model_stage1_best.pth')

pre_model_stage2=os.path.join('pth_stage2','model_stage2_best.pth')

save_path="pth"
save_path_stage2="pth_stage2"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

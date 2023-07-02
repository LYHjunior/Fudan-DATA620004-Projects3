import torch
from configs.config_task2 import get_cfg_defaults
from data.dataset import load_cifar_dataset
from model.ViT import VisionTransformer
from torchmetrics.functional import accuracy, precision_recall,f1_score
from tqdm import tqdm


cfg = get_cfg_defaults()
cfg.TRAIN.batch_size = 3

# load data
train_dataset,test_dataset,num_classes = load_cifar_dataset(cfg)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=2)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=2)

model = VisionTransformer(cfg, zero_head=True, num_classes=100)

# baseline
names=['baseline','cutout','cutmix','mixup']
for name in names:
    model.load_state_dict(torch.load(cfg.BASE_PATH+f'trained_model/{name}/best_model.pt'))
    model = model.to('cuda:0')

    # test
    model.eval()
    preds = []
    targets = []
    for images, labels in tqdm(test_loader):
        images = images.to('cuda:0')
        labels = labels.to('cuda:0')

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        preds.append(pred)
        targets.append(labels)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy(preds,targets)
    pre,recall = precision_recall(preds, targets, average='macro', num_classes=num_classes)
    f1 = f1_score(preds,targets,num_classes = num_classes)
    print('-'*20)
    print(f"{name}:")
    print(f"accuracy is {acc.item()},precision is {pre.item()},recall is {recall.item()},f1 is {f1.item()}")
# trainstage1.py
import torch,argparse,os
import net,config,loaddataset
from tensorboardX import SummaryWriter

# train stage one
def train(args,summary):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset=loaddataset.PreDataset(root='./cifar100/',
                                         train=True,
                                         transform=config.train_transform,
                                         download=True)
    train_data=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=16 , drop_last=True)

    model =net.SimCLRStage1().to(DEVICE)
    lossLR=net.Loss().to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    bestloss = 10000.0
    for epoch in range(1,args.max_epoch+1):
        print('----train-epoch: '+str(epoch))
        model.train()
        total_loss = 0
        for batch,(imgL,imgR,labels) in enumerate(train_data):
            imgL,imgR,labels=imgL.to(DEVICE),imgR.to(DEVICE),labels.to(DEVICE)

            _, pre_L=model(imgL)
            _, pre_R=model(imgR)

            loss=lossLR(pre_L,pre_R,args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        #print("epoch loss:",total_loss/len(train_dataset)*args.batch_size)
        ave = total_loss/len(train_dataset)*args.batch_size
        summary.add_scalar('epochloss_in_train', ave, epoch)

        with open(os.path.join(config.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*args.batch_size) + " ")

        if ave<bestloss:
            bestloss =ave
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage1_best' + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--max_epoch', default=500, type=int, help='')

    args = parser.parse_args()
    logpath = './log_task1new/log--real/'
    summary = SummaryWriter(log_dir=logpath, comment='')

    
    train(args,summary)
    summary.close()

import time
# import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import SRResNet
from datasets import SRDataset
from utils import *

# 强制使用特定CUDA计算引擎
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试


# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 32  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 8  # number of residual blocks

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 1  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e6  # number of training iterations
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
# 梯度裁剪
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cudnn.benchmark = True
torch.backends.cudnn.enabled = False

def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # large_kernel_size = 9
        # small_kernel_size = 3
        # n_channels = 32
        # n_blocks = 8
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')
    print(f"train_dataset：{train_dataset}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=False)  # note that we're passing the collate function here
    # pin_memory=False,  # 禁用锁页内存
    print(f"train_loader：{train_loader}")
    # Total number of epochs to train for

    # epochs = int(iterations // len(train_loader) + 1)
    epochs=100

    print(f"epochs：{epochs}")
    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        print("开始训练》。。。")
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   'checkpoint_srresnet.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    # nn.MSELoss().to(device)
    :param optimizer: optimizer Adam
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        '''
        (lr_imgs, hr_imgs)：

            这是数据解包操作
            
            自定义 SRDataset 的 __getitem__ 方法返回一对图像 (低分辨率, 高分辨率)
            
            lr_imgs = 当前批次的低分辨率图像
            
            hr_imgs = 当前批次的高分辨率图像


'''
        # print(f"lr_imgs：{lr_imgs},hr_imgs：{hr_imgs}")
        data_time.update(time.time() - start)

        # Move to default device
        # 这里的低分辨率是高分辨率图像的1/4
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # 更新参数

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time, loss=losses))

    #   删除变量 释放空间
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()

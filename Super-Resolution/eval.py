'''
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = './'
test_data_names = ["Set5", "Set14", "BSD100"]

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet

# 加载生成器（生成器本质就是SRResNet）
srgan_generator = torch.load(srgan_checkpoint, weights_only=False)['generator'].to(device)
srgan_generator.eval()
model = srgan_generator

# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=4,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # pin_memory=True)

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                           data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                         data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    # Print average PSNR and SSIM
    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

print("\n")
'''

from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = './'
test_data_names = ["Set5", "Set14", "BSD100"]

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# 加载生成器
srgan_generator = torch.load(srgan_checkpoint, weights_only=False)['generator'].to(device)
srgan_generator.eval()
model = srgan_generator

# 用于保存第一张图像的变量
first_lr = first_sr = first_hr = None

# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=4,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)

            # 如果是第一个测试集的第一张图像，保存结果
            if test_data_name == test_data_names[0] and i == 0:
                # 转换图像到PIL格式以便显示
                # first_lr = convert_image(lr_imgs.squeeze(0), source='imagenet-norm', target='pil')
                first_sr = convert_image(sr_imgs.squeeze(0), source='[-1, 1]', target='pil')
                first_hr = convert_image(hr_imgs.squeeze(0), source='[-1, 1]', target='pil')

            # 计算PSNR和SSIM（保持原有逻辑）
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

# 可视化第一张测试图像
if first_sr and first_hr:
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(np.array(first_sr))
    axes[0].set_title("Super Resolution")
    axes[0].axis('off')

    axes[1].imshow(np.array(first_hr))
    axes[1].set_title("High Resolution")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('first_image_comparison.png', bbox_inches='tight')
    plt.show()
else:
    print("未找到测试图像")

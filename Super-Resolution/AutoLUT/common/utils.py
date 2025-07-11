from loguru import logger
import sys

import cv2
import numpy as np
import torch
from scipy import signal
from tqdm import tqdm


def logger_info(debug, log_path='default_logger.log'):
    logger.remove()
    logger.add(log_path, level='INFO', enqueue=True, context='spawn')
    if not debug:
        logger.add(lambda msg: tqdm.write(msg, end=""), level='DEBUG', enqueue=True, context='spawn', colorize=True)
    else:
        logger.add(sys.stdout, level='DEBUG', enqueue=True, context='spawn')
        logger.info("Log level set to DEBUG")

    return logger


def modcrop(image, modulo):
    if len(image.shape) == 2:
        sz = image.shape
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1]]
    elif image.shape[2] == 3:
        sz = image.shape[0:2]
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1], :]
    else:
        raise NotImplementedError
    return image


def _rgb2ycbcr(img, maxVal=255):
    # O = torch.tensor([[16],
    #               [128],
    #               [128]]).to(img.device)
    # T = torch.tensor([[0.256788235294118, 0.504129411764706, 0.097905882352941],
    #               [-0.148223529411765, -0.290992156862745, 0.439215686274510],
    #               [0.439215686274510, -0.367788235294118, -0.071427450980392]]).to(img.device)
    #
    # if maxVal == 1:
    #     O = O / 255.0
    #
    # t = torch.reshape(img, (img.shape[0] * img.shape[1], img.shape[2])).to(T.dtype)
    # t = torch.matmul(t, torch.transpose(T,0,1))
    # t[:, 0] += O[0]
    # t[:, 1] += O[1]
    # t[:, 2] += O[2]
    # ycbcr = torch.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    #
    # return ycbcr

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr


def PSNR(y_true, y_pred, shave_border=4):
    """
       计算峰值信噪比:
       pred: 预测图像 (Y通道)
       target: 真实图像 (Y通道)
       scale: 超分比例
       """
    print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")  # y_true: (168, 168), y_pred: (512, 512)
    # 确保输入是 NumPy 数组（如果是张量则转换）
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    # 转换为 float32 类型（避免整数运算溢出）
    target_data = y_true.astype(np.float32)
    ref_data = y_pred.astype(np.float32)
    print(
        f"target_data: {target_data.shape}, ref_data: {ref_data.shape}")  # target_data: (168, 168), ref_data: (512, 512)

    # 计算 MSE 和 PSNR
    diff = ref_data - target_data
    # 裁剪边缘
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]

    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(255. / rmse)

    '''
    为什么用Y通道：人眼对亮度变化最敏感
    
    为什么用PSNR：超分辨率领域标准评估指标
    
    单位：分贝(dB)，值越高表示质量越好
    '''


def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim

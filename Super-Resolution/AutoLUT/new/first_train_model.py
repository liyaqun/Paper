# 参数
import argparse
import os
import random
import sys
import time
import torch.nn.functional as F
from tqdm import tqdm, trange
from flatbuffers.builder import np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from AutoLUT.MuLUT_AutoLUT.common.option import BaseOptions
import torch.optim as optim
from loguru import logger
import torch
from AutoLUT.MuLUT_AutoLUT.new import model



def logger_info(debug, log_path='default_logger.log'):
    logger.remove()
    logger.add(log_path, level='INFO', enqueue=True, context='spawn')
    if not debug:
        logger.add(lambda msg: tqdm.write(msg, end=""), level='DEBUG', enqueue=True, context='spawn', colorize=True)
    else:
        logger.add(sys.stdout, level='DEBUG', enqueue=True, context='spawn')
        logger.info("Log level set to DEBUG")

    return logger


# DIV2K 数据集
class DIV2K(Dataset):
    # data_class(scale, path, patch_size, debug=debug, gpuNum=gpuNum) 调用
    def __init__(self, scale, path, patch_size, rigid_aug=True, debug=False, gpuNum=1):

        super(DIV2K, self).__init__()
        self.scale = scale  # 4
        self.sz = patch_size  # 48*48
        self.rigid_aug = rigid_aug  # True
        self.path = path  # 训练集文件夹路径
        # 文件列表：['0001',  '0002' ..................]
        # self.file_list = [str(i).zfill(4)
        #                   for i in range(1, 901)]  # use both train and valid
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 401)]
        self.debug = debug  # False
        self.gpuNum = gpuNum  # 1

        if not debug and gpuNum == 1:
            # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
            self.hr_cache = os.path.join(path, "cache_hr.npy")
            print(f"缓存的路径为：{self.hr_cache}")
            if not os.path.exists(self.hr_cache):
                self.cache_hr()
                logger.info("HR image cache to: {}", self.hr_cache)
            #   数据加载
            self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
            # {'0001': array([[[69, 59, 59],
            #         [67, 56, 62],
            #         [74, 62, 68],
            #  ...
            #         [22, 12,  0],
            #         [22, 12,  2],
            #         [22, 12,  3]]], dtype=uint8), '0002': array([[[ 90, 110, 160],
            #         [ 90, 110, 160],
            #         [ 90, 110, 160],
            #         ...,
            print(f"self.hr_ims: {self.hr_ims}")

            logger.info("HR image cache from: {}", self.hr_cache)

            # 加载低分辨率数据集
            self.lr_cache = os.path.join(path, "cache_lr_x{}.npy".format(self.scale))
            if not os.path.exists(self.lr_cache):
                self.cache_lr()
                logger.info("LR image cache to: {}", self.lr_cache)
            self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
            logger.info("LR image cache from: {}", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        print(f"低清图像的路径：{dataLR}")
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f + "x{}.png".format(self.scale))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        print(f"高清图像的路径：{dataHR}")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f + ".png")))
        print(f"hr_dict:{hr_dict}")
        # {'0001': array([[[69, 59, 59],
        #         [67, 56, 62],
        #         [74, 62, 68],
        #         ...,
        #
        #         [22, 12,  0],
        #         [22, 12,  2],
        #         [22, 12,  3]]], dtype=uint8),
        #  '0002': array([[[ 90, 110, 160],
        #         [ 90, 110, 160],
        #         [ 90, 110, 160],
        #      ...............
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        # 随机选择一个文件

        key = random.choice(self.file_list)
        print(f"随机选择的key:{key}")
        if self.debug or self.gpuNum > 1:
            dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
            im = np.array(Image.open(os.path.join(dataLR, key + "x{}.png".format(self.scale))))
            dataHR = os.path.join(self.path, "HR")
            lb = np.array(Image.open(os.path.join(dataHR, key + ".png")))
        else:
            # 获取到lb、im :这张图片key的详细数据
            lb = self.hr_ims[key]
            im = self.lr_ims[key]
            print(f"lb shape: {lb.shape}, im shape: {im.shape}")
        #     随机选择的key:0002
        #     lb shape: (1404, 2040, 3), im shape: (339, 510, 3)  # 3代表3通道

        shape = im.shape  # (339, 510, 3)  低分辨率图像
        # self.sz ： 裁剪的图像大小：48
        i = random.randint(0, shape[0] - self.sz)  # 从0到 图像宽-48  随机产生一个数i
        j = random.randint(0, shape[1] - self.sz)  # 从0到 图像高-48  随机产生一个数i
        c = random.choice([0, 1, 2])  # 从0 1 2随机选择一个数值
        print(f"lb:{lb}")
        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, c]
        # 裁剪后的HR块：192×192像素 (48×4=192) --> 可以仔细琢磨琢磨 这里不会越界
        im = im[i:i + self.sz, j:j + self.sz, c]
        # 裁剪后的LR块：48×48像素

        # 最终得到的是单通道(c)的图像块

        '''
        这段代码执行的是数据增强（Data Augmentation）操作，特别是针对超分辨率任务的刚性变换增强。
        目的是通过对训练数据进行随机变换，增加数据多样性，提高模型的泛化能力。
        '''
        # rigid_aug：True
        if self.rigid_aug:  # 如果启用图像增强
            #  # 水平翻转 (50%概率)
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)
            #  # 垂直翻转 (50%概率)
            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            # # 随机旋转 (0°, 90°, 180°或270°)
            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        # 数据类型转换：astype(np.float32)：将uint8图像转为float32  便于归一化计算
        # 归一化： / 255.0 --->   将像素值从[0, 255]  --->   缩放到[0, 1]范围
        # 添加通道维度：np.expand_dims(..., axis=0)
        # 将形状从(H, W)  -->   变为(1, H, W)   【符合PyTorch的输入格式要求：(通道数, 高度, 宽度)】

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)  # High resolution
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)  # Low resolution

        print(f"处理后的数据：{lb},{im}")
        # [[[0.10588235 0.10980392 0.10196079 ... 0.33333334 0.3137255  0.34509805]
        #   [0.10980392 0.12941177 0.10980392 ... 0.34509805 0.45490196 0.40392157]
        print(f"处理后的数据的形状：{lb.shape},{im.shape}")
        # 处理后的数据的形状：(1, 192, 192),(1, 48, 48)
        return im, lb  # Input, Ground truth

    def __len__(self):
        return 400


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size, debug=False, gpuNum=1, data_class=DIV2K,
                 # 图像的个数
                 length=400):
        # batch_size= 32 num_workers：8 scale：4  path（trainDir）：训练文件夹  cropSize（patch_size）：图像裁剪大小：48*48  data_class：DIV2K数据集  length：数据集的长度

        # 高分辨率图像和低分辨率图像
        self.data = data_class(scale, path, patch_size, debug=debug, gpuNum=gpuNum)
        print(f"data是:{self.data}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.length = length

        self.is_cuda = True
        self.build()
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.length

    def build(self):
        self.data_iter = iter(DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        ))

    def next(self):
        try:

            batch = next(self.data_iter)
            self.iteration += 1
            print(f"batch[0], batch[1]:{batch[0], batch[1]}")
            return batch[0], batch[1]
        # 这里的图像数量一定要大于等于batch_size的数量 不然到不了一个批次
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            print(f"batch[0], batch[1]:{batch[0], batch[1]}")
            return batch[0], batch[1]


class DebugDataProvider:
    def __init__(self, dataset):
        self.dataset = dataset
        self.build()
        self.iteration = 0
        self.epoch = 0

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.dataset))

    def next(self):
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            return batch


# 学习率动态设置
class LrLambda:
    def __init__(self, opt):
        self.set5_psnr = 0
        self.opt = opt

    def __call__(self, i):
        # # 根据PSNR调整学习率
        if self.set5_psnr < 20:
            #   # 根据PSNR水平调整验证间隔  结果越不好，就不要多次进行验证  当结果比较好了  再进行验证
            self.opt.valStep = 10
            return 10  # 高学习率
        elif self.set5_psnr < 24:
            self.opt.valStep = 20
            return 1
        elif self.set5_psnr < 28.1:
            self.opt.valStep = 30
            return 0.5
        elif self.set5_psnr < 30:
            self.opt.valStep = 50
            return 0.1
        else:
            return 0.05  # 低学习率


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # data
        parser.add_argument('--batchSize', type=int, default=5)
        parser.add_argument('--cropSize', type=int, default=48, help='input LR training patch size')
        parser.add_argument('--trainDir', type=str, default="D:/OneDrive/桌面/ML/data/train/DIV2K_train_HR")
        parser.add_argument('--valDir', type=str, default='D:/OneDrive/桌面/ML/data/valid/DIV2K_valid_HR')

        # training
        parser.add_argument('--startIter', type=int, default=0,
                            help='Set 0 for from scratch, else will load saved params and trains further')
        parser.add_argument('--totalIter', type=int, default=20000, help='Total number of training iterations')
        parser.add_argument('--displayStep', type=int, default=100, help='display info every N iteration')
        parser.add_argument('--valStep', type=int, default=2000, help='validate every N iteration')
        # valStep : 每几次循环 就进行一次验证
        parser.add_argument('--saveStep', type=int, default=200, help='save models every N iteration')
        parser.add_argument('--lr0', type=float, default=1e-3)
        parser.add_argument('--lambda-lr', action=argparse.BooleanOptionalAction)
        parser.add_argument('--weightDecay', type=float, default=0)
        parser.add_argument('--gpuNum', '-g', type=int, default=1)
        parser.add_argument('--workerNum', '-n', type=int, default=0)  # 不开启多进程
        parser.add_argument('--valServerMode', default=False, action='store_true')
        parser.add_argument('--valServer', type=str, default='none',
                            help="When in validate server mode, sets the bind addr. Otherwise, set the addr of the validating server. ")
        parser.add_argument('--gradientStep', type=int, default=250,
                            help='Write gradients into tensorboard every N iteration')

        self.isTrain = True
        return parser

    def process(self, opt):
        return opt


def train(opt, logger, rank=0):
    #  初始化组件: 日志文件：expDir='../models\\debug\\expr_1'
    writer = SummaryWriter(log_dir=opt.expDir)

    stages = opt.stages

    # opt.model='SRNets'
    model_class = getattr(model, opt.model)

    model_G = model_class(opt.numSamplers, opt.sampleSize, nf=opt.nf, scale=opt.scale, stages=stages,
                          act=opt.activateFunction, ).cuda()

    # 优化器
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    print(f"params_G:{params_G}")

    # lr would be multiplied by lr_lambda(i)
    #  parser.add_argument('--lr0', type=float, default=1e-3)
    #  parser.add_argument('--lambda-lr', action=argparse.BooleanOptionalAction)
    # 初始化学习率
    initial_lr = 1 if opt.lambda_lr else opt.lr0
    # print(f"initial_lr:{initial_lr}") # 0.001

    # Adam 优化器和学习率  # weightDecay=0
    opt_G = optim.Adam(params_G, lr=initial_lr, weight_decay=opt.weightDecay, amsgrad=False, fused=True)

    # # 动态学习率
    print(f"opt.lambda_lr:{opt.lambda_lr}")  # opt.lambda_lr:None
    if opt.lambda_lr:
        print("# 动态学习率")
        lr_sched_obj = LrLambda(opt)
        lr_sched = optim.lr_scheduler.LambdaLR(opt_G, lr_sched_obj)

    # 数据加载器
    # 训练集 : trainDir='D:/OneDrive/桌面/ML/data/train/train2017'  COCO的数据集
    # scale=4
    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize, debug=opt.debug,
                          gpuNum=opt.gpuNum)

    # train_iter 返回的是？？？？？？？
    # if opt.debug:
    #     train_iter = DebugDataProvider(DebugDataset())

    # 验证集
    # valid = SRBenchmark(opt.valDir, scale=opt.scale)

    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # startIter=0
    i = opt.startIter

    # 训练主循环
    # totalIter=200

    with trange(opt.startIter + 1, opt.totalIter + 1, dynamic_ncols=True) as pbar:
        for i in pbar:
            # 开始训练
            print(f"开始训练....")
            model_G.train()

            # Data preparing
            st = time.time()

            # 数据加载 --> 低分辨率图像  高分辨率图像
            im, lb = train_iter.next()
            print(f"im:{im.shape},lb:{lb.shape}")  # 低分辨率图像  高分辨率图像
            # im:torch.Size([1, 1, 48, 48]),lb:torch.Size([1, 1, 192, 192])
            # return batch[0], batch[1]

            im = im.to(rank)  # 低分辨率图像
            lb = lb.to(rank)  # 高分辨率真实标签

            dT += time.time() - st

            # TRAIN G
            st = time.time()
            opt_G.zero_grad()

            pred = model_G.forward(im, 'train')

            loss_G = F.mse_loss(pred, lb)  # 损失计算:mse
            loss_G.backward()  # 反向传播
            opt_G.step()

            #  学习率调整
            if opt.lambda_lr:
                lr_sched.step()
                pbar.set_postfix(lr=lr_sched.get_last_lr(), val_step=lr_sched_obj.opt.valStep)

            rT += time.time() - st

            # For monitoring
            accum_samples += opt.batchSize
            l_accum[0] += loss_G.item()

            # 日志记录
            if i % opt.displayStep == 0 and rank == 0:
                writer.add_scalar('loss_Pixel', l_accum[0] / opt.displayStep, i)
                if opt.lambda_lr:
                    writer.add_scalar('learning_rate', torch.tensor(lr_sched.get_last_lr()), i)

                l_accum = [0., 0., 0.]
                dT = 0.
                rT = 0.

            #  模型保存
            if i % opt.saveStep == 0:
               if opt.gpuNum == 1:
                    SaveCheckpoint(model_G, opt_G, lr_sched if opt.lambda_lr else None, opt, i, logger)

            # 验证
            # target_val_step = opt.valStep if opt.lambda_lr == False else lr_sched_obj.opt.valStep
            # # 进行验证的频率
            # if i % target_val_step == 0:
            #     # validation during multi GPU training
            #     if opt.gpuNum > 1:
            #         set5_psnr = valid_steps(model_G.module, valid, opt, i, writer, rank, logger)
            #     else:
            #         if opt.valServer.lower() == "none":
            #             set5_psnr = valid_steps(model_G, valid, opt, i, writer, rank, logger)
            #         else:
            #             send_val_request('http://' + opt.valServer, model_G, i, logger)
            #     if opt.lambda_lr:
            #         lr_sched_obj.set5_psnr = set5_psnr
            #
            # # Display gradients
            # if i % opt.gradientStep == 0:
            #     for name, param in model_G.named_parameters():
            #         writer.add_histogram(f'gradients/{name}', param.grad, i)

    logger.info("Complete")


def SaveCheckpoint(model_G, opt_G, Lr_sched_G, opt, i, logger, best=False):
    print(f"模型保存.....")
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, os.path.join(
        opt.expDir, 'Model_{:06d}{}.pth'.format(i, str_best)))
    torch.save(opt_G, os.path.join(
        opt.expDir, 'Opt_{:06d}{}.pth'.format(i, str_best)))
    if Lr_sched_G != None:
        torch.save(Lr_sched_G, os.path.join(
            opt.expDir, 'LRSched_{:06d}.pth'.format(i)))

    logger.info("Checkpoint saved {}".format(str(i)))


def main():
    print(f"参数加载....")
    opt_inst = TrainOptions()  # 参数解析器
    opt = opt_inst.parse()  # 解析命令行参数
    # print(f"opt:{opt}")
    # opt:Namespace(model='SRNets', task='sr', scale=4, sigma=25, qf=20, nf=64, stages=2, sampleSize=3, numSamplers=3, activateFunction='relu', interval=4, modelRoot='../models', expDir='../models\\debug\\expr_3', load_from_opt_file=False, debug=False, batchSize=32, cropSize=48, trainDir='D:/OneDrive/桌面/ML/data/train/DIV2K_train_HR', valDir='D:/OneDrive/桌面/ML/data/valid/DIV2K_valid_HR', startIter=0, totalIter=200, displayStep=100, valStep=2000, saveStep=200, lr0=0.001, lambda_lr=None, weightDecay=0, gpuNum=1, workerNum=8, valServerMode=False, valServer='none', gradientStep=250, isTrain=True, modelDir='../models\\debug', modelPath='../models\\debug\\expr_3\\Model.pth', valoutDir='../models\\debug\\expr_3\\val')
    # 存储日志信息
    logger = logger_info(opt.debug, os.path.join(opt.expDir, 'train-{time}.log'))
    print(logger)
    train(opt, logger)


if __name__ == '__main__':
    main()

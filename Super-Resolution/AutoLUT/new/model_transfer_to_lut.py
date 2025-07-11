import os

import torch

from AutoLUT.MuLUT_AutoLUT.common.option import BaseOptions
import model
from AutoLUT.MuLUT_AutoLUT.new.network import SRNet, AutoSample, MuLUTUnit
import numpy as np


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--loadIter', '-i', type=int, default=20000)
        parser.add_argument('--testDir', type=str, default='D:/OneDrive/桌面/ML/data')  # 测试数据的文件夹
        parser.add_argument('--resultRoot', type=str, default='../results')
        parser.add_argument('--lutName', type=str, default='LUT')

        self.isTrain = False
        return parser


'''
生成覆盖所有可能输入组合的张量，用于构建完整的查找表:生成所有可能的2x2输入块组合
'''


def get_input_tensor(opt):
    # 1D输入（构造1D的lut表）
    # interval=4
    base = torch.arange(0, 257, 2 ** opt.interval)  # 从 0-256 ，间隔为2^interval  共17个类别
    # tensor([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256])
    # base[-1] -= 1
    L = base.size(0)  # 拿到上面的长度 ： 17

    # 2D 输入（构造2D的lut表） -->  两个通道的所有组合
    # 即如下所示：
    '''
    tensor([[0, 0],
            [0, 16],
            [0, 32],
            [0, 48],
            [0, 64],
            [0, 80],
            [0, 96],
            [0, 112],
            [0, 128],
            [0, 144],
            [0, 160],
            [0, 176],
            [0, 192],
            [0, 208],
            [0, 224],
            [0, 240],
            [0, 256],
            [16, 0],
            [16, 16],
            [16, 32],
            [16, 48],
            [16, 64],
            [16, 80],
            [16, 96],
            [16, 112],
            [16, 128],
            [16, 144],
            [16, 160],
            [16, 176],
            [16, 192],
            [16, 208],
            [16, 224],
            [16, 240],
            [16, 256],
            ..................
           
    '''
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D 输入（构造3D的lut表） -->  三个通道的任意所有组合
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    # 即：
    '''
    [[  0,   0,   0],
    [  0,   0,  16],
    [  0,   0,  32],
        ...,
    [256, 256, 224],
    [256, 256, 240],
    [256, 256, 256]]
    '''
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)
    '''
    [[  0,   0,   0,   0],
        [  0,   0,   0,  16],
        [  0,   0,   0,  32],
        ...,
        [256, 256, 256, 224],
        [256, 256, 256, 240],
        [256, 256, 256, 256]],
    '''
    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 2, 2).float() / 255.0
    '''
    [[[[0.0000, 0.0000],
          [0.0000, 0.0000]]],
        [[[0.0000, 0.0000],
          [0.0000, 0.0627]]],
    '''
    # 将各像素值除以255
    return input_tensor


'''
将训练好的SRNets模型（包含多个SRNet模块）转换为查找表（LUT）格式，并保存相关的权重（如自适应采样器的权重和残差权重）
'''
if __name__ == "__main__":
    opt_inst = TestOptions()
    opt = opt_inst.parse()
    # TestOptions: Namespace(model='SRNets', task='sr', scale=4, sigma=25, qf=20, nf=64, stages=2, sampleSize=3, numSamplers=3, activateFunction='relu', interval=4, modelRoot='../models', modes='sdy', expDir='../models\\debug\\expr_9', load_from_opt_file=False, debug=False, loadIter=20000, testDir='D:/OneDrive/桌面/ML/data', resultRoot='../results', lutName='LUT', isTrain=False, flag=4, modelDir='../models\\debug', modelPath='../models\\debug\\expr_9\\Model.pth')
    print(f"TestOptions: {opt}")

    model_class = getattr(model, opt.model)

    model_G: model.SRNets = model_class(nf=opt.nf, scale=opt.scale, stages=opt.stages, num_samplers=opt.numSamplers,
                                        sample_size=opt.sampleSize, act=opt.activateFunction).cuda()

    #  加载模型：  加载训练好的权重
    lm = torch.load(os.path.join("D:/pycharm/LUT/LUT/AutoLUT/MuLUT_AutoLUT/models/debug/expr_5",
                                 'Model_{:06d}.pth'.format(opt.loadIter)), weights_only=False)
    print(f"lm:{lm.state_dict()}")
    model_G.load_state_dict(lm.state_dict(), strict=True)

    # # 遍历所有阶段和采样器
    for stage in range(opt.stages):
        # 两个阶段
        for sampler in range(opt.numSamplers):
            # 三个采样
            input_tensor = get_input_tensor(opt)
            print(f"LUT表：{input_tensor}")
            # Split input to not over GPU memory

            # input_tensor.size() ： torch.Size([83521, 1, 2, 2]) --->  N = (256/2^interval)^4
            # 也就是4D的LUT 该张量有 83521 个样本，1 个通道，2 行和 2 列。
            # 2*2 大小块 输入batch：83521

            #  # 分批处理避免内存溢出 每次输入835个像素
            B = input_tensor.size(0) // 100
            outputs = []

            # Extract the corresponding part from network
            name = f"s{stage}_{sampler}"
            print(f"___________{name}___________")

            # # 获取当前stage和sampler对应的模块
            srnet: SRNet = getattr(model_G, name)
            auto_sampler: AutoSample = srnet.sampler

            # 保存自适应采样器权重
            # weight shape: [out_channels, in_channels, kernel_size, kernel_size]
            raw_sw = auto_sampler.sampler.weight
            sampler_weights = raw_sw.cpu().detach().numpy()
            # check_weights(sampler_weights, auto_sampler.input_shape)
            # 自适应采样器的卷积核权重
            sampler_path = os.path.join(opt.expDir,
                                        f"LUT_x{opt.scale}_{opt.interval}bit_int8_s{stage}_{sampler}_sampler.npy")
            np.save(sampler_path, sampler_weights)

            #  # 保存残差权重（如果存在）
            mulutunit: MuLUTUnit = srnet.model

            if mulutunit.has_residual:
                residual_wegiths = mulutunit.residual.weights.cpu().detach().numpy()
                # 残差连接的融合权重
                residual_path = os.path.join(opt.expDir,
                                             f"LUT_x{opt.scale}_{opt.interval}bit_int8_s{stage}_{sampler}_residual.npy")
                np.save(residual_path, residual_wegiths)

            # 禁用残差连接（LUT生成时不使用）
            mulutunit.has_residual = False

            # 生成查找表内容
            with torch.no_grad():
                model_G.eval()  # 设置为评估模式
                for b in range(100 + 1):  # 分批处理 分为100批 没批835个
                    batch_input = input_tensor[b * B:(b + 1) * B] # 切分
                    # batch_input：torch.Size([21, 1, 2, 2])
                    # 记录网络输出：对每个输入组合（固定区间代表像素）执行前向传播
                    # 输入：2x2低分辨率块
                    # 输出：上采样后的高分辨率块（如4x4）
                    batch_output = mulutunit(batch_input)  # # 前向传播
                    # 得到一个这个2*2低分辨率块 对应的 上采样后的高分辨率块
                    # 这样就可以直接查表，在低分辨率图像中找到其像素数值A，再往LUT表中找提前存好的低分辨率A对应的高分辨率B，返回即可
                    print(f"mulutunit操作后的batch_input: {batch_input}")

                    assert (-1 <= batch_output).all() and (
                            batch_output <= 1).all(), f"Output out of bound: {batch_output}"

                    # 量化为int8格式（-128到127）
                    results = torch.round(torch.clamp(batch_output, -1, 1)
                                          * 127).cpu().data.numpy().astype(np.int8)
                    print(f"结果是“{results}")
                    outputs += [results]

            #  # 合并并保存LUT
            results = np.concatenate(outputs, 0)


            lut_path = os.path.join(opt.expDir,
                                    "LUT_x{}_{}bit_int8_s{}_{}.npy".format(opt.scale, opt.interval, stage, sampler))
            np.save(lut_path, results)

            # 一共就6种组合方式/模式  迭代6次
            print(f"Conv layers ({results.shape}) -> {lut_path}, sampler ({sampler_weights.shape}) -> {sampler_path}")

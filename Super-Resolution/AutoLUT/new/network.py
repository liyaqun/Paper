import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from loguru import logger
from rich import inspect


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()  # 计算每个参数的元素数量
    print(net)  # 打印网络结构
    print('total number of parameters: %.3f K' % (num_params / 1e3))


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


############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        ## MSRA初始化 (He初始化)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            #  # 偏置初始化为0
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, act='relu'):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64, act='relu'):
        super(DenseConv, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        # 与原始输入相接
        out = torch.cat([x, feat], dim=1)
        return out


############### Other tools ###############

class Residual(nn.Module):
    #  input_shape = (2, 2)
    def __init__(self, input_shape):
        assert len(input_shape) == 2
        super().__init__()
        self.shape = input_shape
        self.weights = nn.Parameter(torch.zeros(self.shape))

    def forward(self, x, prev_x):
        assert x.shape[-2:] == self.shape and prev_x.shape[-2:] == self.shape
        with torch.no_grad():
            self.weights.data = torch.clamp(self.weights, 0, 1)
        #
        averaged = self.weights * prev_x + (1 - self.weights) * x

        return averaged


class AutoSample(nn.Module):

    # input_size =3 （采样比例）
    def __init__(self, input_size: int):
        print(f"AutoSample开始了...")
        super().__init__()
        self.input_shape = input_size
        # 卷积 3 * 3 卷积 -> 输入通道由1通道转为4通道
        self.sampler = nn.Conv2d(1, 4, input_size)
        # 生成扩大2倍 的 图像（拼接）
        self.shuffel = nn.PixelShuffle(2)
        self.nw = input_size ** 2
        # print(f"self.nw:{self.nw}")

    def forward(self, x):
        print(f"初始的低分辨率图像大小:{x.shape}")  # 初始的低分辨率图像大小:torch.Size([2304, 1, 3, 3])
        # 输入的是im :　４８＊４８的低分辨率图像（裁剪过后的）
        # print(f"AutoSample:{x}")
        assert len(x.shape) == 4 and x.shape[-2:] == (
            self.input_shape, self.input_shape), f"Unexpected shape: {x.shape}"
        # x = self.sampler(x)
        # logger.debug(self.sampler.weight)
        '''
        self.sampler.weight是卷积层的权重参数，通过 view 操作将其重塑为二维矩阵
        使用 softmax 函数对每一行进行归一化，确保权重和为 1
        归一化后的权重重新 reshape 回原始形状，用于后续卷积操作
        '''
        w = F.softmax(self.sampler.weight.view(-1, self.nw), dim=1).view_as(self.sampler.weight)
        '''
        使用学习到的归一化权重进行卷积操作
        由于权重已经归一化，输出结果会保持在合理范围内
        偏置设为 0，确保只有学习到的权重影响采样结果
        '''
        print(f"权重参数的形状：{w.shape}")
        # x: 低分辨率的图像数值输入  w: ｓｏｆｔｍａｘ后的权重
        # 卷积
        x = F.conv2d(x, w, bias=self.sampler.bias * 0)
        print(f"卷积后的x的形状：{x.shape}")  # torch.Size([2304, 4, 1, 1])  ：nn.Conv2d(1, 4, input_size)
        # 输入块3x3 → 卷积核3x3 → 输出特征图1x1
        #  # 使用PixelShuffle进行上采样
        '''
        使用 PixelShuffle 将通道维度的信息重组到空间维度(每个像素所有通道组合拼接)
        实现特征图的上采样，同时保留了自适应采样的信息
        '''
        x = self.shuffel(x)
        print(f"自动采样后的 x.shape:{x.shape}")  # torch.Size([2304, 1, 2, 2])
        return x


'''
mode：决定输入形状和处理方式（'2x2' 是最常用模式）

nf：特征通道数（默认为64）

upscale：上采样比例（超分辨率关键参数）

dense：是否使用密集连接（特征复用增强）

residual：是否使用残差连接（稳定训练）

act：激活函数选择（ReLU或GELU）
'''


############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    #  self.model = MuLUTUnit('2x2', nf, upscale, dense=dense, residual=residual, act=act)
    # nf=64
    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True, residual=False, act='relu'):
        # residual ： 是否使用残差模块看不同的stage的模式【前面会传入状态】
        # nf： 64 输出通道
        super(MuLUTUnit, self).__init__()
        #  设置激活函数
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        # 上采样因子与残差连接标志
        self.upscale = upscale
        self.has_residual = residual

        #  # 输入形状与初始卷积层
        if mode == '2x2':  # 只执行到这
            self.input_shape = (2, 2)
            self.conv1 = Conv(1, nf, 2)  # 输出通道64 将 1通道的2x2输入 转换为 nf通道的特征表示
        elif mode == '2x2d':
            self.input_shape = (3, 3)
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.input_shape = (4, 4)
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.input_shape = (1, 4)
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if residual:
            # Residual模块：学习权重融合当前输入和前级输入
            self.residual = Residual(self.input_shape)

        # # 密集连接卷积块（核心特征提取部分）
        if dense:
            # 每层输出 = 当前层特征 + 前面所有层特征拼接 特征复用率：100%（所有前面特征都传递到后续层）
            self.conv2 = DenseConv(nf, nf, act=act)
            self.conv3 = DenseConv(nf + nf * 1, nf, act=act)
            self.conv4 = DenseConv(nf + nf * 2, nf, act=act)
            self.conv5 = DenseConv(nf + nf * 3, nf, act=act)
            self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)  # 输出卷积层
        else:
            self.conv2 = ActConv(nf, nf, 1, act=act)
            self.conv3 = ActConv(nf, nf, 1, act=act)
            self.conv4 = ActConv(nf, nf, 1, act=act)
            self.conv5 = ActConv(nf, nf, 1, act=act)
            # upscale ： 两种选择：1 or 4
            self.conv6 = Conv(nf, upscale * upscale, 1)

        # 上采样
        if self.upscale > 1:
            # 将通道维度重组为空间维度
            # 例如：16通道 → 重组为4x4空间分辨率（4倍上采样）
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x, prev_x=None):
        print(f"输入到MuLUT的数据x的形状：{x.shape}")  # torch.Size([2304, 1, 2, 2])

        if self.has_residual:
            x = self.residual(x, prev_x)
        x = self.act(self.conv1(x))  # [块数, 64, 1, 1] 输入2*2形状 2*2卷积之后的特征图为1*1
        x = self.conv2(x)  # [B, 128, 1, 1]
        x = self.conv3(x)  # [B, 192, 1, 1]
        x = self.conv4(x)  # [B, 256, 1, 1]
        x = self.conv5(x)  # [B, 320, 1, 1]
        print(f"conv5处理后的x的形状：{x.shape}")
        x = torch.tanh(self.conv6(x))  # # [B, 16, 1, 1] (4x上采样时)
        print(f"MULUT处理后的x的形状：{x.shape}")
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        print(f"输出MuLUT的数据x的形状：{x.shape}")
        return x

#
# class MuLUTcUnit(nn.Module):
#     """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """
#
#     def __init__(self, mode, nf, act='relu'):
#         super(MuLUTcUnit, self).__init__()
#         if act == 'relu':
#             self.act = nn.ReLU()
#         elif act == 'gelu':
#             self.act = nn.GELU()
#         else:
#             raise AttributeError(f"Unknown activate function: {act}")
#
#         if mode == '1x1':
#             self.conv1 = Conv(3, nf, 1)
#         else:
#             raise AttributeError
#
#         self.conv2 = DenseConv(nf, nf)
#         self.conv3 = DenseConv(nf + nf * 1, nf)
#         self.conv4 = DenseConv(nf + nf * 2, nf)
#         self.conv5 = DenseConv(nf + nf * 3, nf)
#         self.conv6 = Conv(nf * 5, 3, 1)
#
#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = torch.tanh(self.conv6(x))
#         return x
#

############### Image Super-Resolution ###############
class SRNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    # numSamplers=3, sampleSize=3
    def __init__(self, sample_size, nf=64, upscale=1, dense=True, residual=False, act='relu'):
        # sample_size ： 3
        # upscale ： 分阶段：第一阶段：2  第二阶段：1
        # print(sample_size, nf, upscale, dense, residual)
        super(SRNet, self).__init__()
        self.residual = residual

        self.K = sample_size  # 采样块大小 (3x3)
        self.S = upscale  # 上采样比例：1
        print(f"SRNet开始了....")
        self.sampler = AutoSample(sample_size)  # 自适应采样器

        # MuLUTUnit 网络结构 ：  # LUT核心单元
        # 经过这个网络的输出结果：([2304, 1, 4, 4]) 或者 ([2304, 1, 1, 1])
        self.model = MuLUTUnit('2x2', nf, upscale, dense=dense, residual=residual, act=act)

        self.P = self.K - 1

    #     self.P=2

    def unfold(self, x):
        # 调用unfold处理padding后的图像：  # 输入[1,1,50,50]

        """
        Do the convolution sampling
        """
        if x is None: return x, None
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L  # 使用F.unfold提取3x3块（kernel_size=3）
        x = x.view(B, C, self.K * self.K,
                   (H - self.P) * (W - self.P))  # B,C,K*K,L  # 块数量计算：(50-3+1) * (50-3+1) = 48*48 = 2304
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        # 输出形状：[2304, 1, 3, 3]
        return x, (B, C, H, W)

    def put_back(self, x, ori_shape):
        # AutoSample 和 MuLUTUnit 的输出会通过put_back方法重新组合为完整图像
        # 维度重组：将分块处理的结果（形状为[B*C*L, K, K]）恢复为适合fold操作的格式
        # 特征图合并：使用F.fold将局部特征图合并为全局特征图
        # 尺寸匹配：通过self.S参数（上采样因子）确保输出尺寸与目标分辨率一致

        B, C, H, W = ori_shape
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)

        # torch.Size([1, 1, 48, 48]) 或者 torch.Size([1, 1, 192, 192])
        print(f"AutoSample 和 MuLUTUnit 的输出 经过维度重组后的x形状：{x.shape}")
        return x

    def forward(self, x, prev_x=None):
        print(f"SRNet输入的x数据：{x}")
        # Here, prev_x is unfolded multiple times (previously unfolded as x)

        # logger.debug(f"SRNet got {x.shape}")
        x, shape = self.unfold(x)  # 生成图像块 [2304, 1, 3, 3]
        print(f"低分辨率图像经过unfold处理：{shape}")
        prev_x, _ = self.unfold(prev_x)  # if x is None: return x, None
        # 此时prev_x == x

        # 将unfold后的数据传入采样器： 输入形状 torch.Size([2304, 1, 3, 3])
        print(f"将unfold处理后的图像 输入到采样器：")
        x = self.sampler(x)  # 传入AutoSample
        # logger.debug(f"after sample {x}")
        if prev_x is not None:
            prev_x = self.sampler(prev_x)

        print(f"自动采样后x的形状结果：{x.shape},{prev_x}")  # torch.Size([2304, 1, 2, 2])
        # x | prev_x：经过自动采样和亚像素混洗后的图像数值（tensor张量）

        x = self.model(x, prev_x)  # B*C*L,K,K
        # 返回的x的形状:[2304, 1, 1, 1]) 或者 [2304, 1, 4, 4]
        # logger.debug(f"shape after model: {x.shape}")

        x = self.put_back(x, shape)

        return x

#
# ############### Grayscale Denoising, Deblocking, Color Image Denosing ###############
# class DNNet(nn.Module):
#     """ Wrapper of basic MuLUT block without upsampling. """
#
#     def __init__(self, mode, nf=64, dense=True):
#         super(DNNet, self).__init__()
#         self.mode = mode
#
#         self.S = 1
#         if mode == 'Sx1':
#             self.model = MuLUTUnit('2x2', nf, dense=dense)
#             self.K = 2
#         elif mode == 'Dx1':
#             self.model = MuLUTUnit('2x2d', nf, dense=dense)
#             self.K = 3
#         elif mode == 'Yx1':
#             self.model = MuLUTUnit('1x4', nf, dense=dense)
#             self.K = 3
#         else:
#             raise AttributeError
#         self.P = self.K - 1
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = F.unfold(x, self.K)  # B,C*K*K,L
#         x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
#         x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
#         x = x.reshape(B * C * (H - self.P) * (W - self.P),
#                       self.K, self.K)  # B*C*L,K,K
#         x = x.unsqueeze(1)  # B*C*L,l,K,K
#
#         if 'Y' in self.mode:
#             x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
#                            x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
#
#             x = x.unsqueeze(1).unsqueeze(1)
#
#         x = self.model(x)  # B*C*L,K,K
#         x = x.squeeze(1)
#         x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
#         x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
#         x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
#         x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
#                    self.S, stride=self.S)
#         return x

#
# ############### Image Demosaicking ###############
# class DMNet(nn.Module):
#     """ Wrapper of the first stage of MuLUT network for demosaicking. 4D(RGGB) bayer patter to (4*3)RGB"""
#
#     def __init__(self, mode, nf=64, dense=False):
#         super(DMNet, self).__init__()
#         self.mode = mode
#
#         if mode == 'SxN':
#             self.model = MuLUTUnit('2x2', nf, upscale=2, out_c=3, dense=dense)
#             self.K = 2
#             self.C = 3
#         else:
#             raise AttributeError
#         self.P = 0  # no need to add padding self.K - 1
#         self.S = 2  # upscale=2, stride=2
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         # bayer pattern, stride = 2
#         x = F.unfold(x, self.K, stride=2)  # B,C*K*K,L
#         x = x.view(B, C, self.K * self.K, (H // 2) * (W // 2))  # stride = 2
#         x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
#         x = x.reshape(B * C * (H // 2) * (W // 2),
#                       self.K, self.K)  # B*C*L,K,K
#         x = x.unsqueeze(1)  # B*C*L,l,K,K
#
#         # print("in", torch.round(x[0, 0]*255))
#
#         if 'Y' in self.mode:
#             x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
#                            x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
#
#             x = x.unsqueeze(1).unsqueeze(1)
#
#         x = self.model(x)  # B*C*L,out_C,S,S
#         # self.C along with feat scale
#         x = x.reshape(B, C, (H // 2) * (W // 2), -1)  # B,C,L,out_C*S*S
#         x = x.permute((0, 1, 3, 2))  # B,C,outC*S*S,L
#         x = x.reshape(B, -1, (H // 2) * (W // 2))  # B,C*out_C*S*S,L
#         x = F.fold(x, ((H // 2) * self.S, (W // 2) * self.S),
#                    self.S, stride=self.S)
#         return x

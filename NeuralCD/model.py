import torch
import torch.nn as nn


# 评估学生对知识点的掌握程度和预测答题正确率
class Net(nn.Module):
    '''
    NeuralCDM:神经网络认知诊断模型
    '''

    # net = Net(student_n, exer_n, knowledge_n)
    def __init__(self, student_n, exer_n, knowledge_n):
        # 知识点的数量
        self.knowledge_dim = knowledge_n
        # 练习题的数量
        self.exer_n = exer_n
        # 学生数量
        self.emb_num = student_n
        # # 学生嵌入维度（等于知识点数量）
        self.stu_dim = self.knowledge_dim
        # # 预测网络输入维度（等于知识点数量）
        self.prednet_input_len = self.knowledge_dim
        #  预测网络两个隐藏层的节点数
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(Net, self).__init__()

        # 网络架构
        #  # 学生嵌入层：将学生ID映射为掌握程度向量  stu_dim：维度就是列数
        # self.emb_num：行数:学生数量/ID    列数:一个学生是self.stu_dim 维的向量  -->列数是知识点的数量
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        # 知识点难度层：将练习题ID映射为知识点难度向量
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # # 题目区分度层：将练习题ID映射为区分度标量
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        #   # 预测网络的三层全连接结构
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        # 最后只输出一个概率值
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                #  # 使用Xavier正态分布初始化权重
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''

        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


'''
遍历指定的网络层（prednet_full1, prednet_full2, prednet_full3）

对每层的权重矩阵：

找出所有负值

将这些负值设置为0（加上它们的绝对值）
'''


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))  # 计算负权重的绝对值
            w.add_(a)

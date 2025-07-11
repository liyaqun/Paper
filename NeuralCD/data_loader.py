import json
import torch


class TrainDataLoader(object):

    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []

        data_file = 'data/train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        # print(f"文本数据：{self.data}")
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        # print(f"知识的数量：{knowledge_n}")
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        # print(f"nextbatch:,,,,,,")
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        # 一次读取32个batch
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            # print(f"初始化：{knowledge_emb}") # 一维0向量
            for knowledge_code in log['knowledge_code']:
                # 如果存在这个知识 就把这个知识写为1
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            # 存在这个学生 就加入数组中
            input_stu_ids.append(log['user_id'] - 1)
            # 存在这个练习题 就加入数组中
            input_exer_ids.append(log['exer_id'] - 1)
            # 将知识向量加入数组中
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
        #     将成绩加入数组中

        # 更换起始点
        self.ptr += self.batch_size
        # print(f"input_stu_ids：{input_stu_ids}")
        # print(f"input_exer_ids：{input_exer_ids}")
        # print(f"input_knowedge_embs：{input_knowedge_embs}")
        # 以张量的形式返回
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        # 读取到末尾结束
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            data_file = 'data/val_set.json'
        else:
            data_file = 'data/test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

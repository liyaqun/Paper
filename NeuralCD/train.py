import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net



# 4163 名学生、17746 个练习题和 123个知识概念，
exer_n = 17746
knowledge_n = 123
student_n = 4163

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
# 5轮循环
epoch_n = 5


def train():
    data_loader = TrainDataLoader()

    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')


    # 换一个交叉熵损失函数
    loss_function = nn.NLLLoss()


    for epoch in range(epoch_n):
        # 每个epoch都会使用全部的训练数据，不会遗漏后面的数据--> while not data_loader.is_end():
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs) # # 模型预测的正确概率 (batch_size, 1)
            output_0 = torch.ones(output_1.size()).to(device) - output_1 # # 计算错误概率（1 - 答对概率）
            # 为啥要拼接：是为了适配负对数似然损失函数（NLLLoss）的输入要求
            output = torch.cat((output_0, output_1), 1) # 拼接成两列


            # labels 是成绩
            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()

            # 这个函数调用应用了自定义的权重裁剪器，确保预测网络（PredNet）的权重保持非负
            net.apply_clipper() #在每次参数更新（optimizer.step()）后立即调用

            running_loss += loss.item()
            if batch_count % 200 == 199:

                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        # 保存模型
        save_snapshot(net, 'model/model_epoch' + str(epoch))


# 在训练集上测试 并进行评估
def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # 预测正确的样本
    accuracy = correct_count / exer_count
    # rmse评估
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # auc计算
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

# 认知诊断模型（CDM）的扩展版本（NeuralCDM+，或CNCD-Q）
if __name__ == '__main__':
    # sys.argv :
    # ['D:/PyCharm 2023.3.4/plugins/python/helpers/pydev/pydevconsole.py',
    #  '--mode=client',
    #  '--host=127.0.0.1',
    #  '--port=65098']
    # if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
    #     # len(sys.argv) = 4
    #     print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
    #     exit(1)
    # else:
    #     device = torch.device(sys.argv[1])
    #     epoch_n = int(sys.argv[2])

    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train()

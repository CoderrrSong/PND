import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import tqdm


# MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out


# 数据集类
class PatentDataset(Dataset):
    def __init__(self, data_file, pt_folder):
        self.data = self._read_data(data_file)
        self.pt_folder = pt_folder

    def _read_data(self, data_file):
        patent_data = []
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')[:-2]
                doc = line.strip().split(',')[-2]
                label = int(line.strip().split(',')[-1])  # 将标签转换为整数
                patent_data.append((parts, doc, label))
        return patent_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patent_numbers, doc, label = self.data[idx]
        features = []

        # 待检测文档list处理
        for patent_number in patent_numbers:
            #print(patent_number)
            pt_file = self.pt_folder + patent_number + '.pt'
            data = torch.load(pt_file, map_location=torch.device('cpu'))
            length =4096
            title_embed = torch.zeros(1, length)
            abstract_embed = torch.zeros(1, length)
            description_embed = torch.zeros(1, length)
            claims_embed = torch.zeros(1, length)


            # 在数据处理部分将 numpy 数组转换为 PyTorch 张量
            if len(data['title_embed']) != 0:
                title_embed = data['title_embed'].reshape(1, length)

            if len(data['abstract_embed']) != 0:
                abstract_embed = data['abstract_embed'].reshape(1, length)

            if len(data['description_embed']) != 0:
                description_embed =data['description_embed'].reshape(1,length)

            if len(data['claims_embed']) != 0:
                claims_embed = data['claims_embed'].reshape(1, length)

            feature = torch.cat((title_embed, abstract_embed, description_embed, claims_embed))
            features.append(feature)

        features = torch.stack(features)
        features = torch.mean(features, dim=0)

        # 检测位置文档处理
        pt_file_d = self.pt_folder + doc + '.pt'
        data_d = torch.load(pt_file_d, map_location=torch.device('cpu'))

        title_embed_d = torch.zeros(1, length)
        abstract_embed_d = torch.zeros(1, length)
        description_embed_d = torch.zeros(1, length)
        claims_embed_d = torch.zeros(1, length)

        if len(data_d['title_embed']) != 0:
            title_embed_d = data_d['title_embed'].reshape(1, length)

        if len(data_d['abstract_embed']) != 0:
            abstract_embed_d =data_d['abstract_embed'].reshape(1, length)

        if len(data_d['description_embed']) != 0:
            description_embed_d = data_d['description_embed'].reshape(1, length)

        if len(data_d['claims_embed']) != 0:
            claims_embed_d = data_d['claims_embed'].reshape(1, length)

        feature_d = torch.cat((title_embed_d, abstract_embed_d, description_embed_d, claims_embed_d))

        feature_all = torch.cat((features, feature_d))
        f_num, dim = feature_all.size()
        feature_all = feature_all.reshape(f_num * dim)

        # 找出 NaN 值的位置
        mask = torch.isnan(feature_all)
        # 使用 mask 将 NaN 值替换为0
        feature_all = torch.where(mask, torch.zeros_like(feature_all), feature_all)

        # 找出 inf 值的位置
        mask = torch.isinf(feature_all)
        # 使用 mask 将 inf 值替换为0
        feature_all = torch.where(mask, torch.zeros_like(feature_all), feature_all)

        return feature_all, label


# 训练函数
# 训练函数
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, log_file,test_log_file):
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Valid_Loss,Accuracy,Precision,Recall,F1,AUC\n")

    with open(test_log_file, 'w') as f:
        f.write("Epoch,Test_Loss,Accuracy,Precision,Recall,F1,AUC\n")
    for epoch in tqdm.tqdm(range(num_epochs)):
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)  # 多类别交叉熵损失
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # 每一轮epoch结束后进行测试
        valid_loss, accuracy, precision, recall, f1, auc = test(model, valid_loader)
        # 将结果保存到文件中
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{running_train_loss},{valid_loss},{accuracy},{precision},{recall},{f1},{auc}\n")

        # 每一轮epoch结束后进行测试
        test_loss, accuracy, precision, recall, f1, auc = test(model, test_loader)
        # 将结果保存到文件中
        with open(test_log_file, 'a') as f:
            f.write(f"{epoch + 1},{test_loss},{accuracy},{precision},{recall},{f1},{auc}\n")


# 测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels_list = []
    predicted_probabilities = []
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            # 使用 softmax 将输出转换为概率分布
            probabilities = nn.functional.softmax(outputs, dim=1)
            predicted_probabilities.extend(probabilities.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    test_loss = running_test_loss / len(test_loader.dataset)
    accuracy = correct / total
    precision = precision_score(labels_list, predictions, average='macro')
    recall = recall_score(labels_list, predictions, average='macro')
    f1 = f1_score(labels_list, predictions, average='macro')

    # 使用概率分布计算 ROC AUC
    auc = roc_auc_score(labels_list, predicted_probabilities, multi_class='ovr')

    return test_loss, accuracy, precision, recall, f1, auc

if __name__ == '__main__':
    E=100
    Accuracy_list=[]
    Precision_list=[]
    Recall_list=[]
    F1_list=[]
    AUC_list=[]
    for cnt in range(1):
        # 加载数据
        type="all"
        foder=f"D:/Draw/GLM/{type}/"
        train_data = PatentDataset(foder+'train_triple.txt', foder+'fields/')
        print(len(train_data))
        valid_data = PatentDataset(foder+'valid_triple.txt', foder+'fields/')
        print(len(valid_data))
        test_data = PatentDataset(foder+'test_triple.txt', foder+'fields/')
        print(len(test_data))

        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=64)
        test_loader = DataLoader(test_data, batch_size=64)

        # 模型参数
        input_size = 4096 * 8
        hidden_size = 512
        output_size = 3  # 三元组分类任务

        # 初始化模型、损失函数和优化器
        model = MLP(input_size, hidden_size, output_size)
        criterion = nn.CrossEntropyLoss()  # 多类别交叉熵损失
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        test_log_file = f'test_log_triple_{type}.txt'
        # 训练模型并保存结果到文件
        train_log_file = f'train_log_triple_{type}.txt'
        train(model, train_loader, test_loader, criterion, optimizer, num_epochs=E, log_file=train_log_file,test_log_file=test_log_file)



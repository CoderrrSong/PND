import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import tqdm


# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out


# Dataset class
class PatentDataset(Dataset):
    def __init__(self, data_file, pt_folder):
        self.data = self._read_data(data_file)
        self.pt_folder=pt_folder

    def _read_data(self, data_file):
        patent_data = []
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')[:-2]
                doc = line.strip().split(',')[-2]
                label=line.strip().split(',')[-1]

                # Each element contains a list of patent numbers, a document ID, and a label
                patent_data.append((parts, doc, int(label)))
                #print(patent_data)
            return patent_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patent_numbers, doc, label = self.data[idx]
        features = []

        ############################################# Processing the list of documents to be checked #######################################
        for patent_number in patent_numbers:
            #print(patent_numbers)
            pt_file = self.pt_folder + patent_number + '.pt'


            data = torch.load(pt_file,map_location=torch.device('cpu'))

            # Part for initializing feature vectors
            length=4096
            title_embed = torch.zeros(1,length)
            abstract_embed =torch.zeros(1,length)
            description_embed =torch.zeros(1,length)
            claims_embed = torch.zeros(1,length)

            # Convert numpy arrays to PyTorch tensors during data processing
            if len(data['title_embed'])!=0:
                title_embed = data['title_embed'].reshape(1,length)

            if len(data['abstract_embed'])!=0:
                abstract_embed = data['abstract_embed'].reshape(1,length)

            if len(data['description_embed'])!=0:
                description_embed = data['description_embed'].reshape(1,length)

            if len(data['claims_embed'])!=0:
                claims_embed = data['claims_embed'].reshape(1,length)


            # print(title_embed.size())
            # print(abstract_embed.size())
            # print(description_embed.size())
            # print(claims_embed.size())
            # Concatenate non-empty embedding vectors to construct the feature vector
            feature = torch.cat((title_embed, abstract_embed, description_embed, claims_embed))

            features.append(feature)
        features = torch.stack(features)
        features  = torch.mean(features , dim=0)


        ############################################# Processing of documents at the detection position #######################################
        pt_file_d = self.pt_folder + doc + '.pt'
        data_d = torch.load(pt_file_d, map_location=torch.device('cpu'))

        # Section for initializing the feature vector
        title_embed_d = torch.zeros(1,length)
        abstract_embed_d = torch.zeros(1,length)
        description_embed_d =torch.zeros(1,length)
        claims_embed_d = torch.zeros(1,length)

        if len(data_d['title_embed'])!=0:
            title_embed_d =data_d['title_embed'].reshape(1,length)

        if len(data_d['abstract_embed'])!=0:
            abstract_embed_d =data_d['abstract_embed'].reshape(1,length)

        if len(data_d['description_embed'] )!=0:
            description_embed_d =data_d['description_embed'].reshape(1,length)

        if len(data_d['claims_embed'])!=0:
            claims_embed_d = data_d['claims_embed'].reshape(1,length)


        feature_d = torch.cat((title_embed_d, abstract_embed_d, description_embed_d, claims_embed_d))


        ############################################ processing of the documents to be checked and the documents at the detection position #######################################
        # feature_all=torch.cat((features,feature_d))
        feature_all = torch.cat((features, feature_d))
        f_num, dim = feature_all.size()
        feature_all=feature_all.reshape(f_num * dim)

        # Find the positions of NaN values
        mask = torch.isnan(feature_all)
        # Replace NaN values with 0 using a mask
        feature_all = torch.where(mask, torch.zeros_like(feature_all), feature_all)

        # Find the positions of inf values
        mask = torch.isinf(feature_all)
        # Replace inf values with 0 using a mask
        feature_all = torch.where(mask, torch.zeros_like(feature_all), feature_all)

        return feature_all, label


# train
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, log_file,test_log_file):
    with open(log_file, 'a') as f:
        f.write("Epoch,Train_Loss,Valid_Loss,Accuracy,Precision,Recall,F1,AUC\n")
    with open(test_log_file, 'a') as f:
        f.write("Epoch,Test_Loss,Accuracy,Precision,Recall,F1,AUC\n")

    for epoch in tqdm.tqdm(range(num_epochs)):
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # valid
        valid_loss, accuracy, precision, recall, f1, auc = test(model, valid_loader)

        # save
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{running_train_loss},{valid_loss},{accuracy},{precision},{recall},{f1},{auc}\n")


        # test
        test_loss, accuracy, precision, recall, f1, auc = test(model, test_loader)
        # save
        with open(test_log_file, 'a') as f:
            f.write(f"{epoch + 1},{test_loss},{accuracy},{precision},{recall},{f1},{auc}\n")

# test
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
            loss = criterion(outputs, labels.float().unsqueeze(1))
            running_test_loss += loss.item()
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.float().unsqueeze(1)).sum().item()
            predictions.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            predicted_probabilities.extend(outputs.cpu().numpy())

    test_loss = running_test_loss / len(test_loader.dataset)
    accuracy = correct / total
    precision = precision_score(labels_list, predictions)
    recall = recall_score(labels_list, predictions)
    f1 = f1_score(labels_list, predictions)
    auc = roc_auc_score(labels_list, predicted_probabilities)

    return test_loss, accuracy, precision, recall, f1, auc


if __name__ == '__main__':
    E=100
    Accuracy_list=[]
    Precision_list=[]
    Recall_list=[]
    F1_list=[]
    AUC_list=[]
    for cnt in range(1):
        # load data
        type="all"

        foder=f"D:/Draw/GLM/{type}/"
        train_data = PatentDataset(foder+'train_tuple.txt', foder+'fields/')
        print(len(train_data))
        valid_data = PatentDataset(foder+'valid_tuple.txt', foder+'fields/')
        print(len(valid_data))
        test_data = PatentDataset(foder+'test_tuple.txt', foder+'fields/')
        print(len(test_data))
        # Create data loader
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=64)
        test_loader = DataLoader(test_data, batch_size=64)
        # Model parameters
        input_size =4096 * 8
        hidden_size = 512
        output_size = 1

        # Initialize model, loss function, and optimizer
        model = MLP(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        test_log_file = f'test_log_tuple_{type}.txt'
        # train
        train_log_file = f'train_log_tuple_{type}.txt'
        train(model, train_loader, test_loader, criterion, optimizer, num_epochs=E, log_file=train_log_file,test_log_file=test_log_file)


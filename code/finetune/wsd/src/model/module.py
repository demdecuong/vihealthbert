import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, config, dropout_rate=0.1):
        super().__init__()

        self.dropout_1 = nn.Dropout(dropout_rate*2)
        self.dense_1  = nn.Linear(config.hidden_size*2, 128)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dense_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature):

        feature = self.dropout_1(feature)
        feature = self.dense_1(feature)
        feature = self.relu(feature)
        feature = self.dropout_2(feature)
        feature = self.dense_2(feature).view(-1)

        feature = self.sigmoid(feature)
        return feature

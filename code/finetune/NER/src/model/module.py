import torch.nn as nn

class SlotClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_slot_labels,
        dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        self.num_slot_labels = num_slot_labels
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

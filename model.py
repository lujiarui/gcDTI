"""Graph convolutional network(GCN) model with attention layer.
Implementation by Pytorch and Geometric packages.
Reference:
    - https://pytorch.org/
    - https://github.com/rusty1s/pytorch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCN_CNN(torch.nn.Module):
    """Utilize GCN(with attention layer) to extract features from drugs
        CNN for targets
    """
    def __init__(self, n_output=1, n_feature_xd=78, n_feature_xt=25,
                 n_filter=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GCN_CNN, self).__init__()
        
        # GCN layers with attention layer to do atom convolution
        self.conv1 = GATConv(n_feature_xd, n_feature_xd, heads=10)
        self.conv2 = GCNConv(n_feature_xd * 10, n_feature_xd * 10)
        self.fc_g1 = torch.nn.Linear(n_feature_xd * 10 * 2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(n_feature_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filter, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filter*121, output_dim)

        # Concatenated layer
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)        # n_output = 1 for regression task

    def forward(self, data):
        """Feed forward operation
        x represents for drugs while xt represents for targets
        """
        # unzip data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Apply global max pooling and global mean pooling
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        
        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # Concat
        xc = torch.cat((x, xt), 1)
        # Dense layers
        xc = F.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc2(xc))
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


net = GCN_CNN()
print(net)
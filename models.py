import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class Autoencoder(nn.Module):
    """
    Autoencoder model for feature encoding.

    Args:
        input_dim (int): Dimension of input features.
        encoding_dim (int): Dimension of encoded features.
    """
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, encoding_dim)
        self.fc2 = nn.Linear(encoding_dim, encoding_dim)
        self.fc3 = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class GATWithAutoencoderResidual(nn.Module):
    """
    GAT model with autoencoder and residual connections.

    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output (number of classes).
        encoding_dim (int): Dimension of encoded features.
        heads (int): Number of attention heads.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, encoding_dim, heads=1):
        super(GATWithAutoencoderResidual, self).__init__()
        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        self.gat1 = GATConv(encoding_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.residual_fc1 = nn.Linear(encoding_dim, hidden_dim * heads)
        self.residual_fc2 = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_encoded = self.autoencoder(x)
        x_gat1 = self.gat1(x_encoded, edge_index)
        x_residual1 = self.residual_fc1(x_encoded)
        x = F.relu(x_gat1 + x_residual1)
        x_gat2 = self.gat2(x, edge_index)
        x_residual2 = self.residual_fc2(x)
        x = x_gat2 + x_residual2
        return x
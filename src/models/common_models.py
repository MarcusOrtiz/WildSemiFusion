import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FourierFeatureLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_dim, out_dim) * 2 * torch.pi)

    def forward(self, x):
        return torch.cat([torch.sin(x @ self.weights), torch.cos(x @ self.weights)], dim=-1)

class BlackAndWhiteCNN(nn.Module):
    """
    Acts as black and white image input encoder for the compression layer

    CNNs are better with batch normalization, use it for BW
    """

    def __init__(self):
        super(BlackAndWhiteCNN, self).__init__()
        pass


class ColorCNN(nn.Module):
    """
    Acts as color image input encoder for the compression layer
    - Use LAB
    - Deeper than BlackAndWhiteCNN
    - CNNs are better with batch normalization, however, first have to understand why colornet uses LayerNorm
    """

    def __init__(self):
        super(ColorCNN, self).__init__()
        pass


class ColorNet(nn.Module):
    """
    Labnet is used to quantize the RGB

    Design Question: Should the expert be trained on logits or softmax? If logits then we can produce a bigger weight
    which may make more sense. It can also be trained on softmax and then in the end to end it will not be connected
    with the softmax layer
    """

    def __init__(self, in_features=512, hidden_dim=256, num_bins=313):
        super(ColorNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, 3 * num_bins)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])  # Flatten for LayerNorm
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, 313)
        x = F.softmax(x, dim=-1)  # Apply Softmax to color bins
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim=512, output_activation=None):
        super(ResidualBlock, self).__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

        self.residual1 = nn.Linear(input_dim, 256)
        self.residual2 = nn.Linear(256, 128)

        self.activation = nn.ReLU()  # Use ReLU for internal activations
        self.output_activation = output_activation  # This can be None, nn.Sigmoid(), nn.Tanh(), etc.
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res1 = self.residual1(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = x + res1
        res2 = self.residual2(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = x + res2
        x = self.layer3(x)

        if self.output_activation:
            x = self.output_activation(x)

        return x


class SemanticNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=11):
        super(SemanticNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape the input for BatchNorm1d if necessary
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))  # Flatten batch and locations if needed

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return x
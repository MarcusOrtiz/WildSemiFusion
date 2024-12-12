import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureLayer(nn.Module):
    def __init__(self, in_dim=2, out_dim=256):
        super(FourierFeatureLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_dim, out_dim // 2) * 2 * torch.pi)

    def forward(self, x):
        return torch.cat([torch.sin(x @ self.weights), torch.cos(x @ self.weights)], dim=-1)


class GrayscaleCNN(nn.Module):
    def __init__(self, image_size, out_dim):
        super(GrayscaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool = nn.MaxPool2d(2, 2)  # used at every layer

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32 * (image_size[0] // (2 ** 2)) * (image_size[1] // (2 ** 2)), out_dim)

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn1(self.conv1(x))))  # [batch, 16, H/2, W/2]
        x = self.max_pool(F.relu(self.bn2(self.conv2(x))))  # [batch, 16, H/4, W/4]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        return x


class LABCNN(nn.Module):
    def __init__(self, image_size, out_dim):
        super(LABCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2, 2)  # used at every layer

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * (image_size[0] // (2 ** 3)) * (image_size[1] // (2 ** 3)), out_dim)

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn1(self.conv1(x))))
        x = self.max_pool(F.relu(self.bn2(self.conv2(x))))
        x = self.max_pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        return x


class CompressionLayer(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super(CompressionLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class ColorNet(nn.Module):
    def __init__(self, in_features, hidden_dim, num_bins):
        super(ColorNet, self).__init__()
        self.num_bins = num_bins
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])  # Flatten for LayerNorm
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        return x


class SimpleColorNet(ColorNet):
    def __init__(self, in_features, hidden_dim, num_bins):
        super(SimpleColorNet, self).__init__(in_features, hidden_dim, num_bins)
        self.fc3 = nn.Linear(hidden_dim, 3 * num_bins)

    def forward(self, x):
        x = super(SimpleColorNet, self).forward(x)
        x = self.fc3(x)
        x = x.view(-1, 3, self.num_bins)
        return x


class ComplexColorNet(ColorNet):
    def __init__(self, in_features, hidden_dim_1, hidden_dim_2, hidden_dim_3, num_bins):
        super(ComplexColorNet, self).__init__(in_features, hidden_dim_1, num_bins)

        self.residual1 = nn.Linear(in_features, hidden_dim_1)
        self.residual2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        self.fc3 = nn.Linear(hidden_dim_1 + in_features, hidden_dim_2)
        self.bn3 = nn.LayerNorm(hidden_dim_2)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.bn4 = nn.LayerNorm(hidden_dim_2)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.bn5 = nn.LayerNorm(hidden_dim_2)
        self.dropout5 = nn.Dropout(0.4)

        self.fc6 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn6 = nn.LayerNorm(hidden_dim_3)
        self.dropout6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(hidden_dim_3, 3 * num_bins)

    def forward(self, x):
        residual1 = self.residual1(x)
        x = super(ComplexColorNet, self).forward(x)
        x = torch.cat([x, residual1], dim=-1)

        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)

        residual2 = self.residual2(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        x = x + residual2

        x = F.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01)
        x = self.dropout5(x)
        x = F.leaky_relu(self.bn6(self.fc6(x)), negative_slope=0.01)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = x.view(-1, 3, self.num_bins)
        return x


class SemanticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SemanticNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

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

        return x


class SimpleSemanticNet(SemanticNet):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleSemanticNet, self).__init__(input_dim, hidden_dim, num_classes)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc3(x)
        return x



class ComplexSemanticNet(SemanticNet):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ComplexSemanticNet, self).__init__(input_dim, hidden_dim, num_classes)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = self.fc5(x)
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


class CombineLABColor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CombineLABColor, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, lab1, lab2):
        pass
#       Receive logits for color prediction
#       Split up the lab images to l, a, b
#       Pass through small nueral network to combine each channel
#       Remake the image and return

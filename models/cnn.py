import torch
from torch import nn
from utils import profiles_tensor_to_onehot

class CNN(nn.Module):

    def __init__(
        self,
        max_num_voters: int,
        max_num_alternatives: int,
        architecture_parameters: dict = None,
    ):
        super().__init__()

        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        # Architecture parameters
        architecture_parameters = architecture_parameters or {}
        kernel1 = architecture_parameters.get('kernel1', [3, 1])
        kernel2 = architecture_parameters.get('kernel2', [1, 3])
        channels = architecture_parameters.get('channels', 32)
        hidden_size = architecture_parameters.get('hidden_size', 256)
        layer_norm = architecture_parameters.get('layer_norm', True)  # helps with training stability
        dropout = architecture_parameters.get('dropout', 0.0)

        # Inferring shapes
        K1_0, K1_1 = kernel1
        K2_0, K2_1 = kernel2

        H1_in = max_num_alternatives
        W1_in = max_num_voters
        H1_out = H1_in - (K1_0 - 1)
        W1_out = W1_in - (K1_1 - 1)
        assert H1_out > 0, f'kernel1[0] too large: {kernel1}'
        assert W1_out > 0, f'kernel1[1] too large: {kernel1}'

        H2_out = H1_out - (K2_0 - 1)
        W2_out = W1_out - (K2_1 - 1)
        assert H2_out > 0, f'kernel2[0] too large: {kernel2}'
        assert W2_out > 0, f'kernel2[1] too large: {kernel2}'

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=max_num_alternatives,
            out_channels=channels,
            kernel_size=(K1_0, K1_1),
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(K2_0, K2_1),
            padding=0
        )

        # Flatten 
        self.flatten = nn.Flatten()

        # Fully connected layers  
        fc_in_dim = channels * H2_out * W2_out
        self.fc1 = nn.Linear(fc_in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, max_num_alternatives)
        
        # Normalization
        if layer_norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.norm3 = nn.LayerNorm(hidden_size)
        else:
            self.norm1 = self.norm2 = self.norm3 = nn.Identity()

        # Activation
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input x: [B, V, A]
        Reshaped internally to [B, A, A, V]
        '''
        x = profiles_tensor_to_onehot(x)  # [B, V, A, A], float
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, C=A, H=A, W=V]

        # Convolutions
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.dropout(self.relu(self.norm1(self.fc1(x))))
        x = self.dropout(self.relu(self.norm2(self.fc2(x))))
        x = self.dropout(self.relu(self.norm3(self.fc3(x))))
        x = self.fc4(x) # logits [B, A]
        return x

    def _model2rule(self, profile_tensor, threshold=0.5, full=False):
        logits = self.forward(profile_tensor)
        winners = (torch.sigmoid(logits) > threshold).long()

        # We could filter out invalid alternatives here
        # if not full:
        #     valid_mask = profile_tensor.ne(-1).any(dim=1)  # [B, A]
        #     winners = winners * valid_mask.long()
        
        return winners

    def model2rule(self, threshold=0.5, full=False):
        return lambda profile_tensor: self._model2rule(
            profile_tensor, threshold=threshold, full=full
        )

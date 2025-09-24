import torch
from torch import nn
from utils import profiles_tensor_to_onehot, PAD_PREF

class MLP(nn.Module):
    def __init__(
        self,
        max_num_voters: int,
        max_num_alternatives: int,
        architecture_parameters: dict = None,
    ):
        super().__init__()

        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        architecture_parameters = architecture_parameters or {}
        hidden_size = architecture_parameters.get('hidden_size', 256)
        dropout_rate = architecture_parameters.get('dropout', 0.0)
        use_layer_norm = architecture_parameters.get('layer_norm', True)

        input_dim = max_num_voters * max_num_alternatives * max_num_alternatives

        # Layer blocks with normalization followed by nonlinearity and dropout
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_size, max_num_alternatives)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = profiles_tensor_to_onehot(x)

        # Flatten to [B, V*A*A]
        x = x.view(x.size(0), -1)

        # Forward pass through block layers
        x = self.input_layer(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.hidden_layer(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.hidden_layer2(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.output_layer(x)
        return x
    
    def _model2rule(
        self,
        profile_tensor: torch.Tensor,
        threshold: float = 0.5,
        full: bool = False,
    ) -> torch.Tensor:
        logits = self.forward(profile_tensor)
        winners = (torch.sigmoid(logits) > threshold).long()

        # We could filter out invalid alternatives here
        # if not full:
        #     # Mask out alternatives that never appear in the election
        #     valid_mask = profile_tensor.ne(PAD_PREF).any(dim=1)  # [B, A]
        #     winners = winners * valid_mask.long()

        return winners
    
    def model2rule(self, threshold: float = 0.5, full: bool = False):
        return lambda profile_tensor: self._model2rule(
            profile_tensor, threshold=threshold, full=full
        )

import torch
from torch import nn
from utils import profiles_tensor_to_onehot, PAD_PREF

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, context=None, key_padding_mask=None):
        context = x if context is None else context
        attn_output, _ = self.mha(
            x, context, context,
            key_padding_mask=key_padding_mask
        )
        x = self.ln(x + attn_output)
        ff_output = self.ff(x)
        return self.ln2(x + ff_output)

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_inducing_points):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        # nn.init.xavier_uniform_(self.inducing_points)  # keeps inducing tokens well-scaled for stable attention
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        # I = self.inducing_points.repeat(B, 1, 1)
        I = self.inducing_points.expand(B, -1, -1)  # expand: no copy
        # Use mask when keys are voters
        H = self.mab1(I, x, key_padding_mask=key_padding_mask)
        # Do NOT use mask when keys are inducing points
        return self.mab2(x, H)

class PoolingMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_seeds):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        # nn.init.xavier_uniform_(self.seeds)  # seeds start diverse and stable for pooling attention
        self.mab = MultiheadAttentionBlock(d_model, num_heads)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        # S = self.seeds.repeat(B, 1, 1)
        S = self.seeds.expand(B, -1, -1)  # expand: no copy
        # Use mask since keys are voters
        return self.mab(S, x, key_padding_mask=key_padding_mask)

class SetTransformer(nn.Module):
    def __init__(self, 
                 max_num_voters: int, 
                 max_num_alternatives: int,
                 architecture_parameters: dict = None,
                ):
        super().__init__()
        
        architecture_parameters = architecture_parameters or {}
        d_model = architecture_parameters.get('d_model', 256)
        layer_norm = architecture_parameters.get('layer_norm', True)
        num_heads = architecture_parameters.get('num_heads', 4)
        num_inducing_points = architecture_parameters.get('num_inducing_points', 8)
        num_seeds = architecture_parameters.get('num_seeds', 1)
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives    

        # Embed each voter's ranking
        self.rank_embedding = nn.Linear(max_num_alternatives * max_num_alternatives, d_model)
        
        # Stabilize embeddings and logits
        Norm = nn.LayerNorm if layer_norm else nn.Identity  # this would still create 2 different norm layers
        self.emb_ln  = Norm(d_model)
        self.head_ln = Norm(d_model)

        # One ISAB block
        self.isab = InducedSetAttentionBlock(d_model, num_heads, num_inducing_points)

        # PMA pooling
        self.pma = PoolingMultiheadAttention(d_model, num_heads, num_seeds)

        # Output layer
        self.fc = nn.Linear(d_model, max_num_alternatives)

    def forward(self, x):      

        # Create voter mask for attention: True where padded, False where valid
        key_padding_mask = x.eq(PAD_PREF).all(dim=2)  # [B, V]
        
        x = profiles_tensor_to_onehot(x)  # [B, V, A, A], float
        x = x.view(x.size(0), x.size(1), -1)  # [B, V, A*A]
        
        # Embed each voter: [B, V, d_model]
        x = self.rank_embedding(x)

        # LN after embedding
        x = self.emb_ln(x) 

        # ISAB with masking on voters
        x = self.isab(x, key_padding_mask=key_padding_mask)

        # PMA with masking on voters
        x = self.pma(x, key_padding_mask=key_padding_mask).squeeze(1)
        
        # LN before output head, keeps logits scale stable
        x = self.head_ln(x)

        # Output logits [B, A]
        logits = self.fc(x)
        return logits

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

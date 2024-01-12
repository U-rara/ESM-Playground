import torch
from torch import nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _alpha_from_logits(self, logits, mask_row, mask_col, inf=1e4):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair.bool(), logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row.bool(), alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, input1, mask1, input2, mask2):
        query1 = self._heads(self.query1(input1), self.num_heads, self.head_size)
        key1 = self._heads(self.key1(input1), self.num_heads, self.head_size)
        query2 = self._heads(self.query2(input2), self.num_heads, self.head_size)
        key2 = self._heads(self.key2(input2), self.num_heads, self.head_size)
        logits11 = torch.einsum('blhd, bkhd->blkh', query1, key1)
        logits12 = torch.einsum('blhd, bkhd->blkh', query1, key2)
        logits21 = torch.einsum('blhd, bkhd->blkh', query2, key1)
        logits22 = torch.einsum('blhd, bkhd->blkh', query2, key2)

        alpha11 = self._alpha_from_logits(logits11, mask1, mask1)
        alpha12 = self._alpha_from_logits(logits12, mask1, mask2)
        alpha21 = self._alpha_from_logits(logits21, mask2, mask1)
        alpha22 = self._alpha_from_logits(logits22, mask2, mask2)

        value1 = self._heads(self.value1(input1), self.num_heads, self.head_size)
        value2 = self._heads(self.value2(input2), self.num_heads, self.head_size)
        output1 = (torch.einsum('blkh, bkhd->blhd', alpha11, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha12, value2).flatten(-2)) / 2
        output2 = (torch.einsum('blkh, bkhd->blhd', alpha21, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha22, value2).flatten(-2)) / 2

        return output1, output2


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, num_heads=8, batch_norm=False, activation="relu"):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(CrossAttentionBlock(hidden_dim, num_heads))

        if batch_norm:
            self.protein_batch_norm_layers = nn.ModuleList()
            self.text_batch_norm_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.protein_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
                self.text_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):

        for i, layer in enumerate(self.layers):
            protein_output, text_output = layer(protein_input_ids, protein_attention_mask,
                                                text_input_ids, text_attention_mask)
            if self.batch_norm:
                protein_output = self.protein_batch_norm_layers[i](protein_output.transpose(1, 2)).transpose(1, 2)
                text_output = self.text_batch_norm_layers[i](text_output.transpose(1, 2)).transpose(1, 2)
            if self.activation:
                protein_output = self.activation(protein_output)
                text_output = self.activation(text_output)

        return {
            'protein_output': protein_output,
            'text_output': text_output,
        }

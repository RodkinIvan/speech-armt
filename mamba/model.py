import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from munch import Munch


class FeedForward(nn.Module):
  def __init__(self, n_embed, dropout) -> None:
    super().__init__()
    self.ffn = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed),
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.ffn(x)

class Block(nn.Module):
  def __init__(self, n_embed, dropout, d_state, d_conv, expand, device) -> None:
    super().__init__()
    self.sa_head = Mamba(
      d_model=n_embed,
      d_state=d_state,
      d_conv=d_conv,
      expand=expand,
    ).to(device)
    self.ffn = FeedForward(n_embed, dropout)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)


  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x

class MambaAudioModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.vocab_size = config.vocab_size
    self.n_embed = config.n_embed
    self.block_size = config.block_size
    self.n_layers = config.n_layers
    self.dropout = config.dropout
    self.device = config.device
    self.mamba_d_state = config.mamba_d_state
    self.mamba_d_conv = config.mamba_d_conv
    self.mamba_expand = config.mamba_expand
    self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
    self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
    self.lm_head = nn.Linear(self.n_embed, self.vocab_size)
    self.ffn = FeedForward(self.n_embed, self.dropout)
    self.blocks = nn.Sequential(*[
      Block(
        self.n_embed,
        self.dropout,
        self.mamba_d_state,
        self.mamba_d_conv,
        self.mamba_expand,
        self.device
      )
      for _ in range(self.n_layers)
    ])


  def forward(self, input_ids, labels=None):
    B,T = input_ids.shape
    tok_emb = self.token_embedding_table(input_ids) # (B,T, C_e)
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C_e)
    x = tok_emb + pos_emb # (B,T,Q, C_e)
    x = self.blocks(x) # (B,T,Q, C_e)
    logits = self.lm_head(x) # (B,T,vocab_size)
    loss = None
    if labels is not None:
        # shift logits and labels for autoregressive next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # predict positions 1..T-1
        shift_labels = labels[:, 1:].contiguous()       # true tokens at positions 1..T
        B2, T2, C = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(B2 * T2, C),
            shift_labels.view(-1)
        )
    return Munch(logits=logits, loss=loss)
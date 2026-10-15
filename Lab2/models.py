import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x, mask):
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, width=192, heads=6, layers=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("La dimensione degli embedding deve essere divisibile per il numero di head")
        self.block_size = block_size
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(block_size, width)
        self.blocks = nn.ModuleList(TransformerBlock(width, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))

    def forward(self, tokens):
        length = tokens.shape[1]
        if length > self.block_size:
            raise ValueError("Contesto superiore al block_size")
        x = self.tokens(tokens) + self.positions(torch.arange(length, device=tokens.device))[None]
        for block in self.blocks:
            x = block(x, self.mask[:length, :length])
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, tokens, count, temperature=0.8):
        if temperature <= 0:
            raise ValueError("La temperatura deve essere positiva")
        self.eval()
        for _ in range(count):
            logits = self(tokens[:, -self.block_size:])[:, -1] / temperature
            tokens = torch.cat([tokens, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        return tokens


class TextClassifier(nn.Module):
    def __init__(self, encoder, classes, frozen=True):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        self.encoder.requires_grad_(not frozen)
        self.head = nn.Linear(encoder.config.hidden_size, classes)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(features)


class MultipleChoice(nn.Module):
    def __init__(self, encoder, frozen=False):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        self.encoder.requires_grad_(not frozen)
        self.head = nn.Linear(encoder.config.hidden_size, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, choice_mask, token_type_ids=None):
        batch, choices, length = input_ids.shape
        inputs = dict(input_ids=input_ids.reshape(-1, length), attention_mask=attention_mask.reshape(-1, length))
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids.reshape(-1, length)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.encoder(**inputs).last_hidden_state[:, 0]
        logits = self.head(features).reshape(batch, choices)
        return logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)


class Retriever(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled, dim=1)

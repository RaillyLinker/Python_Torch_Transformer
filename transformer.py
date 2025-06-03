import math
import torch
from torch import nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens):
        return self.embedding(tokens.long()) * self.scale


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pos_embedding', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1, max_len: int = 512):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.position = PositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        token_emb = self.token(x)
        pos_emb = self.position(x)
        return self.dropout(token_emb + pos_emb)


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.float()
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden, feed_forward_hidden, dropout)
        self.input_sublayer = SublayerConnection(hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=512, n_enc_layers=6, heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])

    def forward(self, src_emb, src_mask=None):
        x = src_emb
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadedAttention(heads, d_model, dropout)
        self.cross_attention = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([
            SublayerConnection(d_model, dropout),
            SublayerConnection(d_model, dropout),
            SublayerConnection(d_model, dropout)
        ])

    def forward(self, x, self_mask, enc_output, enc_mask):
        x = self.sublayers[0](x, lambda _x: self.self_attention(_x, _x, _x, mask=self_mask))
        x = self.sublayers[1](x, lambda _x: self.cross_attention(_x, enc_output, enc_output, mask=enc_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=512, n_dec_layers=6, heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, heads, d_ff, dropout) for _ in range(n_dec_layers)
        ])

    def forward(self, dec_emb, tgt_mask, enc_output, src_mask):
        x = dec_emb
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask, enc_output, src_mask)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            vocab_size_src,
            vocab_size_tgt,
            emb_size,
            nhead,
            dim_feedforward,
            dropout,
            max_len,
            pad_token_id
    ):
        super().__init__()
        self.src_emb = InputEmbedding(vocab_size_src, emb_size, dropout, max_len)
        self.tgt_emb = InputEmbedding(vocab_size_tgt, emb_size, dropout, max_len)
        self.encoder = Encoder(emb_size, num_encoder_layers, nhead, dim_feedforward, dropout)
        self.decoder = Decoder(emb_size, num_decoder_layers, nhead, dim_feedforward, dropout)
        self.generator = nn.Linear(emb_size, vocab_size_tgt, bias=False)
        self.pad_token_id = pad_token_id

    def _make_src_mask(self, src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt):
        batch, tgt_len = tgt.size()
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool)).unsqueeze(
            0).unsqueeze(1)
        pad_mask = (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return pad_mask & subsequent_mask

    def forward(self, src, tgt):
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, tgt_mask, memory, src_mask)
        return self.generator(output)

    def encode(self, src, src_mask=None):
        src_emb = self.src_emb(src)
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        tgt_emb = self.tgt_emb(tgt)
        return self.decoder(tgt_emb, tgt_mask, memory, src_mask)

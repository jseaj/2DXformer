import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class Attention(Module):
    def __init__(
            self,
            dim_qk: int,
            dim_v: int,
            dim_head=32,
            heads=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.linear_q = nn.Sequential(
            nn.Linear(dim_qk, dim_inner, bias=False),
            Rearrange('b ... (h d) -> b h ... d', h=heads)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(dim_qk, dim_inner, bias=False),
            Rearrange('b ... (h d) -> b h ... d', h=heads)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(dim_v, dim_inner, bias=False),
            Rearrange('b ... (h d) -> b h ... d', h=heads)
        )

        self.out_proj = nn.Sequential(
            Rearrange('b h ... d -> b ... (h d)'),
            nn.Linear(dim_inner, dim_v, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value):
        assert query.shape[-1] == key.shape[-1]
        assert key.shape[-2] == value.shape[-2]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out)


class ResidualMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.):
        super(ResidualMLP, self).__init__()
        dim_inner = (dim_in + dim_out) // 2 * 3
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, dim_out),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Identity()
        if dim_in != dim_out:
            self.shortcut = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.norm(x + self.shortcut(residual))
        return x


class ExTBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            dim_head=32,
            heads=4,
            attn_dropout=.0,
            ff_mul=4,
            ff_dropout=.0
    ):
        super(ExTBlock, self).__init__()

        self.attn1 = Attention(d_model, d_model, dim_head, heads, attn_dropout)
        self.post_norm1 = nn.LayerNorm(d_model)

        self.attn2 = Attention(d_model, d_model, dim_head, heads, attn_dropout)
        self.post_norm2 = nn.LayerNorm(d_model)

        dim_ff = int(d_model * ff_mul * 2 / 3)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.post_norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        # x.shape = [b, n, v, d]
        x = self.post_norm1(x + self.attn1(x, x, x))

        x = rearrange(x, 'b n v d -> b v n d')
        x = self.post_norm2(x + self.attn2(x, x, x))
        x = rearrange(x, 'b v n d -> b n v d')

        x = self.post_norm3(x + self.ff(x))
        return x


class EnTBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_var: int,
            dim_head=32,
            heads=4,
            attn_dropout=.0,
            ff_mul=4,
            ff_dropout=.0
    ):
        super(EnTBlock, self).__init__()

        self.attn = Attention(d_model, d_model, dim_head, heads, attn_dropout)
        self.post_norm1 = nn.LayerNorm(d_model)

        self.mlp = ResidualMLP(n_var * d_model, d_model)

        dim_ff = int(d_model * ff_mul * 2 / 3)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.post_norm2 = nn.LayerNorm(d_model)

    def forward(self, x_en: Tensor, x_ex: Tensor):
        """
        Args:
            x_en: embedding of endogenous input, shape [b, n, 1, d]
            x_ex: embedding of exogenous input, shape [b, n, v, d]

        Returns:
            x: shape [b, n, 1, d]
        """
        x = x_en

        x = rearrange(x, 'b n v d -> b v n d')
        x = self.post_norm1(x + self.attn(x, x, x))
        x = rearrange(x, 'b v n d -> b n v d')

        x = rearrange(x, 'b n v d -> b n (v d)')
        x_ex = rearrange(x_ex, 'b n v d -> b n (v d)')
        x = torch.cat([x, x_ex], dim=-1)
        x = self.mlp(x)
        x = rearrange(x, 'b n (v d) -> b n v d', v=1)

        x = self.post_norm2(x + self.ff(x))
        return x


class Layer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_var: int,
            dim_head=32,
            heads=4,
            attn_dropout=.0,
            ff_mul=4,
            ff_dropout=.0
    ):
        super(Layer, self).__init__()

        self.exo_net = ExTBlock(
            d_model,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mul=ff_mul,
            ff_dropout=ff_dropout
        )
        self.endo_net = EnTBlock(
            d_model,
            n_var,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mul=ff_mul,
            ff_dropout=ff_dropout
        )

    def forward(self, x_en: Tensor, x_ex: Tensor):
        x_ex = self.exo_net(x_ex)
        x_en = self.endo_net(x_en, x_ex)
        return x_en, x_ex


class Model(Module):
    def __init__(
            self,
            n_var: int,
            history: int,
            horizon: int,
            d_model: int,
            n_layers: int,
            dim_head=32,
            heads=4,
            attn_dropout=0.,
            ff_mult=4,
            ff_dropout=0.,
            emb_dim=16
    ):
        super().__init__()
        self.n_var = n_var
        self.his = history
        self.hor = horizon

        self.en_emb = ResidualMLP(history, d_model)
        self.ex_emb = ResidualMLP(history, d_model)
        self.time_emb = ResidualMLP(history, d_model)

        self.tod_emb = nn.Embedding(144, emb_dim)
        self.doy_emb = nn.Embedding(366, emb_dim)
        self.moy_emb = nn.Embedding(12, emb_dim)
        self._init_weights()

        self.nets = ModuleList([])
        for _ in range(n_layers):
            self.nets.append(
                Layer(
                    d_model,
                    n_var + 3 * emb_dim,
                    dim_head=dim_head,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_mul=ff_mult,
                    ff_dropout=ff_dropout
                )
            )

        self.pred_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, horizon),
            Rearrange('b n v hor -> b hor n v')
        )

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.tod_emb.weight)
        nn.init.kaiming_uniform_(self.doy_emb.weight)
        nn.init.kaiming_uniform_(self.moy_emb.weight)

    def forward(self, x: Tensor):
        # x.shape = [b, t, n, v]
        time_of_day = x[..., -3]
        day_of_year = x[..., -2]
        mon_of_year = x[..., -1]
        t_emb = torch.cat([
            self.tod_emb(time_of_day.long()),
            self.doy_emb(day_of_year.long()),
            self.moy_emb(mon_of_year.long())
        ], dim=-1)

        # x = x[:, :self.his, :, :-3]
        x = x[..., :-3]
        x_en = rearrange(x[..., -1:], 'b t n v1 -> b n v1 t')
        x_ex = rearrange(x[..., :-1], 'b t n v2 -> b n v2 t')
        t_emb = rearrange(t_emb, 'b t n v3 -> b n v3 t')

        x_en = self.en_emb(x_en)
        t_emb = self.time_emb(t_emb)
        x_ex = self.ex_emb(x_ex)
        x_ex = torch.cat([x_ex, t_emb], dim=-2)
        for net in self.nets:
            x_en, x_ex = net(x_en, x_ex)

        pred = self.pred_proj(x_en)
        return pred


if __name__ == '__main__':
    q = torch.randn([24, 10, 36])
    k = torch.randn([24, 8, 36])
    v = torch.randn([24, 8, 12])

    attn = Attention(36, 12)

    out = attn(q, k, v)
    tmp = 1

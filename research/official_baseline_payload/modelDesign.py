import math

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import Linear

NUM_BITS_PER_RE = 8
PAYLOAD_BITS = 144 * NUM_BITS_PER_RE
# Compatibility-only value read by the repository evaluator's optional prefix diagnostics.
MIDDLE_PREFIX_BITS = 924


# ======================================================================================================================
# Shared building blocks
# ======================================================================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, seq_len, dim]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _tfm_encoder(embed_dim, num_heads, num_layers):
    layer = nn.TransformerEncoderLayer(
        embed_dim, num_heads, dim_feedforward=embed_dim * 2, dropout=0.0, batch_first=True
    )
    return nn.TransformerEncoder(layer, num_layers)


# ======================================================================================================================
# Encoder: compress per-UE subband channel into uplink feedback U
# ======================================================================================================================
class Encoder(nn.Module):
    """Build uplink feedback signal U from downlink subband channel H."""

    def __init__(self):
        super().__init__()
        self._num_re = 96
        self._num_sc_per_sb = 48
        self._num_subbands = 144 // self._num_sc_per_sb
        ant_feat = 2 * 16 * 2
        embed_dim = 128

        self._sb_conv = nn.Conv1d(ant_feat, embed_dim, kernel_size=3, padding=1)
        self._pos = PositionalEncoding(self._num_subbands, embed_dim)
        self._tfm = _tfm_encoder(embed_dim, num_heads=4, num_layers=3)
        self._norm = nn.LayerNorm(embed_dim)
        self._fc_out = nn.Linear(embed_dim * self._num_subbands, self._num_re * 2)

    def forward(self, h, snr):
        # h: [b, rx, tx, subc] complex
        x = torch.stack([torch.real(h), torch.imag(h)], dim=-1)  # [b, rx, tx, subc, 2]
        x = rearrange(x, "b rx tx (sb sc) ri -> (b sb) (rx tx ri) sc", sb=self._num_subbands, sc=self._num_sc_per_sb)
        x = self._sb_conv(x).mean(dim=-1)  # [(b sb) embed_dim]
        x = rearrange(x, "(b sb) d -> b sb d", sb=self._num_subbands)  # [b, sb, embed_dim]
        x = self._pos(x)
        x = self._tfm(x)
        x = self._norm(x)
        x = rearrange(x, "b sb d -> b (sb d)")
        x = self._fc_out(x)  # [b, num_re*2]
        u = torch.complex(x[:, : self._num_re], x[:, self._num_re :])
        return u


# ======================================================================================================================
# Decoder
# ======================================================================================================================
class Decoder(nn.Module):
    """Reconstruct one UE's subband H_hat [b, sb, rx, tx] from its own feedback (shared across UEs)."""

    def __init__(
        self,
        num_re=96,
        num_tx_ant=16,
        num_rx_ant=2,
        num_data_subcarriers=144,
        num_sc_per_sb=48,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_subbands = num_data_subcarriers // num_sc_per_sb
        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        seq_len = self._num_subbands
        ant_feat = num_rx_ant * num_tx_ant * 2

        self._fc_in = nn.Linear(num_re * 2, embed_dim * seq_len)
        self._pos = PositionalEncoding(seq_len, embed_dim)
        self._tfm = _tfm_encoder(embed_dim, num_heads, num_layers)
        self._norm = nn.LayerNorm(embed_dim)
        self._fc_out = nn.Linear(embed_dim, ant_feat)

    def forward(self, feedback):
        # feedback: [b, num_re] complex (a single UE)
        fb = torch.cat([torch.real(feedback), torch.imag(feedback)], dim=-1)  # [b, num_re*2]
        x = self._fc_in(fb)
        x = rearrange(x, "b (seq d) -> b seq d", seq=self._num_subbands)  # [b, sb, embed_dim]
        x = self._pos(x)
        x = self._tfm(x)
        x = self._norm(x)
        x = self._fc_out(x)  # [b, sb, ant_feat]
        x = rearrange(x, "b sb (rx tx ri) -> b sb rx tx ri", rx=self._num_rx_ant, tx=self._num_tx_ant, ri=2)
        h_hat = torch.complex(x[..., 0], x[..., 1])  # [b, sb, rx, tx]
        return h_hat


# ======================================================================================================================
# Precoder
# ======================================================================================================================
class Precoder(nn.Module):
    """Generate per-subband precoder W [b, sb, tx, ue] from reconstructed subband H and noise."""

    def __init__(
        self,
        num_ue=2,
        num_tx_ant=16,
        num_rx_ant=2,
        num_data_subcarriers=144,
        num_sc_per_sb=48,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_ue = num_ue
        self._num_tx_ant = num_tx_ant
        self._num_subbands = num_data_subcarriers // num_sc_per_sb
        seq_len = num_ue * self._num_subbands + 1  # +1 for PRC token
        in_feat = num_rx_ant * num_tx_ant * 2 + 1  # +1 for log-noise

        self._embed = nn.Linear(in_feat, embed_dim)
        self._prc_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self._pos = PositionalEncoding(seq_len, embed_dim)
        self._tfm = _tfm_encoder(embed_dim, num_heads, num_layers)
        self._norm = nn.LayerNorm(embed_dim)
        out_dim = self._num_subbands * num_tx_ant * num_ue * 2
        self._fc_out = nn.Linear(embed_dim, out_dim)

    def forward(self, h_hat, no):
        # h_hat: [b, ue, sb, rx, tx] complex ; no: [b, ue] real noise variance
        b = h_hat.shape[0]
        x = torch.stack([torch.real(h_hat), torch.imag(h_hat)], dim=-1)  # [b, ue, sb, rx, tx, 2]
        x = rearrange(x, "b ue sb rx tx ri -> b (ue sb) (rx tx ri)")  # [b, seq-1, ant_feat]
        no_log = torch.log10(no + 1e-9)  # [b, ue]
        no_log = repeat(no_log, "b ue -> b (ue sb) 1", sb=self._num_subbands)
        x = torch.cat([x, no_log], dim=-1)  # [b, seq-1, in_feat]
        x = self._embed(x)  # [b, seq-1, embed_dim]

        prc = self._prc_token.expand(b, -1, -1)  # [b, 1, embed_dim]
        x = torch.cat([prc, x], dim=1)  # [b, seq, embed_dim]
        x = self._pos(x)
        x = self._tfm(x)
        x = self._norm(x)

        prc_out = x[:, 0, :]  # PRC token aggregates all info
        w = self._fc_out(prc_out)  # [b, sb*tx*ue*2]
        w = rearrange(
            w, "b (sb tx ue ri) -> b sb tx ue ri", sb=self._num_subbands, tx=self._num_tx_ant, ue=self._num_ue, ri=2
        )
        w = torch.complex(w[..., 0], w[..., 1])  # [b, sb, tx, ue]

        # energy normalization per (batch, subband) over (tx, ue)
        energy = torch.sum(torch.abs(w) ** 2, dim=(2, 3), keepdim=True)
        w = w / torch.sqrt(energy + 1e-9)
        return w


# ======================================================================================================================
# Transmitter
# ======================================================================================================================
class Transmitter(nn.Module):
    """MU transmission based on feedbacked CSI."""

    def __init__(self):
        super().__init__()
        self._num_tx_ant = 16
        self._num_ue = 2
        self._num_bits_per_re = NUM_BITS_PER_RE
        self._num_data_subcarriers = 144
        self._num_sc_per_sb = 48
        self._num_subbands = self._num_data_subcarriers // self._num_sc_per_sb

        self._mod = nn.ModuleList(
            [nn.Sequential(Linear(self._num_bits_per_re, 64), nn.ReLU(), Linear(64, 2)) for _ in range(self._num_ue)]
        )

        self._decoder = Decoder(
            num_re=96,
            num_tx_ant=self._num_tx_ant,
            num_data_subcarriers=self._num_data_subcarriers,
            num_sc_per_sb=self._num_sc_per_sb,
        )
        self._precoder = Precoder(
            num_ue=self._num_ue,
            num_tx_ant=self._num_tx_ant,
            num_data_subcarriers=self._num_data_subcarriers,
            num_sc_per_sb=self._num_sc_per_sb,
        )

    def _modulate(self, bits, mod_net):
        num_re = self._num_data_subcarriers
        capacity = num_re * self._num_bits_per_re
        if not 0 < PAYLOAD_BITS <= capacity:
            raise ValueError(f"PAYLOAD_BITS must be in [1, {capacity}]")
        active = torch.arange(capacity, device=bits.device) < PAYLOAD_BITS
        b = (bits[:, :capacity] * active.to(bits.dtype)).reshape(bits.shape[0], num_re, self._num_bits_per_re)
        z = mod_net(b)
        z = torch.complex(z[..., 0::2], z[..., 1::2])
        energy = torch.mean(torch.abs(z) ** 2, dim=(1, 2), keepdim=True)
        z = z / torch.sqrt(energy)
        z = z.permute(0, 2, 1)
        return z

    def forward(self, bits_list, feedback_list, snr_dl, return_aux=False):
        batch_size = bits_list[0].shape[0]

        # Modulation
        z_list = [self._modulate(bits_list[i], self._mod[i]) for i in range(self._num_ue)]
        z = torch.cat(z_list, dim=1)  # [b, ue, sc]
        z = z.permute(0, 2, 1)  # [b, sc, ue]

        # Construct precoder based on feedback
        h_hat = torch.stack(
            [self._decoder(feedback_list[i]) for i in range(self._num_ue)], dim=1
        )  # [b, ue, sb, rx, tx]
        no = rearrange(10 ** (-snr_dl / 10.0), "ue b -> b ue")  # [b, ue]
        w_sb = self._precoder(h_hat, no)  # [b, sb, tx, ue]
        w = repeat(w_sb, "b sb tx ue -> b (sb sc) tx ue", sc=self._num_sc_per_sb)  # [b, sc, tx, ue]

        x = torch.matmul(w, z.unsqueeze(-1)).squeeze(-1)  # [b, sc, tx]
        x = x.permute(0, 2, 1)  # [b, tx, sc]

        ctrl_bits = torch.ones((batch_size, 5), dtype=bits_list[0].dtype, device=bits_list[0].device)

        if return_aux:
            return x, ctrl_bits, h_hat, w_sb
        return x, ctrl_bits


# ======================================================================================================================
# Receiver
# ======================================================================================================================
class Receiver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._num_data_subcarriers = 144
        self._num_tx_ant = 16
        self._num_rx_ant = 2
        self._num_bits_per_re = NUM_BITS_PER_RE
        embed_dim = 128
        num_heads = 4
        num_layers = 6

        in_feat = 2 * self._num_rx_ant + 2 * self._num_rx_ant * self._num_tx_ant + 1
        self._embed = nn.Linear(in_feat, embed_dim)
        self._pos = PositionalEncoding(self._num_data_subcarriers, embed_dim)
        self._tfm = _tfm_encoder(embed_dim, num_heads, num_layers)
        self._norm = nn.LayerNorm(embed_dim)
        self._fc_out = nn.Linear(embed_dim, self._num_bits_per_re)

    def forward(self, y, h, ctrl_bits, snr):
        # y: [b, rx, sc] complex ; h: [b, rx, tx, sc] complex ; snr: [b] real (dB)
        b = y.shape[0]
        sc = self._num_data_subcarriers

        y = y.permute(0, 2, 1)  # [b, sc, rx]
        y_feat = torch.cat([torch.real(y), torch.imag(y)], dim=-1)  # [b, sc, 2*rx]

        h = h.permute(0, 3, 1, 2)  # [b, sc, rx, tx]
        h = h.reshape(b, sc, -1)  # [b, sc, rx*tx]
        h_feat = torch.cat([torch.real(h), torch.imag(h)], dim=-1)  # [b, sc, 2*rx*tx]

        no = 10.0 ** (-snr / 10.0)  # [b]
        no_log = torch.log10(no + 1e-9)[:, None, None]  # [b, 1, 1]
        no_feat = no_log.expand(b, sc, 1)  # [b, sc, 1]

        z = torch.cat([y_feat, h_feat, no_feat], dim=-1)  # [b, sc, in_feat]
        z = self._embed(z)  # [b, sc, embed_dim]
        z = self._pos(z)
        z = self._tfm(z)
        z = self._norm(z)
        z = self._fc_out(z)  # [b, sc, num_bits_per_re]

        llr = z.reshape(b, -1)
        return llr[:, :PAYLOAD_BITS]

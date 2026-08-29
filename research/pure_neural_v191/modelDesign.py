from __future__ import annotations

import math

import torch
from torch import nn

NUM_UE = 2
NUM_UL_RE = 96
NUM_DL_SC = 144
NUM_TX = 16
NUM_CTRL = 5
NUM_BITS_PER_SYMBOL = 8
NUM_BITS_PER_UE = NUM_DL_SC * NUM_BITS_PER_SYMBOL
# Compatibility constants used by the repository's diagnostics-capable evaluator.
# The pure-neural baseline always emits all 1152 logits and does not use these gates.
LOW_SNR_THRESHOLD_DB = -20.0
MIDDLE_PREFIX_THRESHOLD_DB = -20.0
MIDDLE_PREFIX_BITS = 924
SNR_EXPERT_EDGES_DB = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
SNR_EXPERT_BOUNDARIES_DB = SNR_EXPERT_EDGES_DB[1:-1]
NUM_EXPERTS = len(SNR_EXPERT_EDGES_DB) - 1


def _expert_indices(snr: torch.Tensor) -> torch.Tensor:
    boundaries = snr.new_tensor(SNR_EXPERT_BOUNDARIES_DB)
    return torch.bucketize(snr.contiguous(), boundaries, right=True).clamp(0, NUM_EXPERTS - 1)


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length: int, width: int):
        super().__init__()
        position = torch.arange(sequence_length).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        encoding = torch.zeros(sequence_length, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pe[:, : values.shape[1]]


def _transformer(width: int, heads: int, layers: int) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=2 * width,
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerEncoder(block, layers)


class EncoderCore(nn.Module):
    """The organizer baseline Encoder, kept deliberately free of physical priors."""

    def __init__(self):
        super().__init__()
        self._num_re = NUM_UL_RE
        self._num_sc_per_sb = 48
        self._num_subbands = NUM_DL_SC // self._num_sc_per_sb
        width = 128
        self._sb_conv = nn.Conv1d(2 * 16 * 2, width, kernel_size=3, padding=1)
        self._pos = PositionalEncoding(self._num_subbands, width)
        self._tfm = _transformer(width, heads=4, layers=3)
        self._norm = nn.LayerNorm(width)
        self._fc_out = nn.Linear(width * self._num_subbands, self._num_re * 2)

    def forward(self, h: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        del snr
        batch = h.shape[0]
        values = torch.stack([h.real, h.imag], dim=-1)
        values = values.reshape(batch, 2, 16, self._num_subbands, self._num_sc_per_sb, 2)
        values = values.permute(0, 3, 1, 2, 5, 4).reshape(
            batch * self._num_subbands,
            2 * 16 * 2,
            self._num_sc_per_sb,
        )
        values = self._sb_conv(values).mean(dim=-1).reshape(batch, self._num_subbands, -1)
        values = self._norm(self._tfm(self._pos(values))).reshape(batch, -1)
        values = self._fc_out(values)
        return torch.complex(values[:, : self._num_re], values[:, self._num_re :])


class DecoderCore(nn.Module):
    def __init__(self):
        super().__init__()
        self._num_subbands = 3
        self._num_rx_ant = 2
        self._num_tx_ant = 16
        width = 128
        self._fc_in = nn.Linear(NUM_UL_RE * 2, width * self._num_subbands)
        self._pos = PositionalEncoding(self._num_subbands, width)
        self._tfm = _transformer(width, heads=4, layers=3)
        self._norm = nn.LayerNorm(width)
        self._fc_out = nn.Linear(width, self._num_rx_ant * self._num_tx_ant * 2)

    def forward(self, feedback: torch.Tensor) -> torch.Tensor:
        batch = feedback.shape[0]
        values = torch.cat([feedback.real, feedback.imag], dim=-1)
        values = self._fc_in(values).reshape(batch, self._num_subbands, -1)
        values = self._norm(self._tfm(self._pos(values)))
        values = self._fc_out(values).reshape(batch, self._num_subbands, 2, 16, 2)
        return torch.complex(values[..., 0], values[..., 1])


class PrecoderCore(nn.Module):
    def __init__(self):
        super().__init__()
        self._num_ue = NUM_UE
        self._num_subbands = 3
        width = 128
        self._embed = nn.Linear(2 * 16 * 2 + 1, width)
        self._prc_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self._pos = PositionalEncoding(self._num_ue * self._num_subbands + 1, width)
        self._tfm = _transformer(width, heads=4, layers=3)
        self._norm = nn.LayerNorm(width)
        self._fc_out = nn.Linear(width, self._num_subbands * NUM_TX * self._num_ue * 2)

    def forward(self, h_hat: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch = h_hat.shape[0]
        values = torch.stack([h_hat.real, h_hat.imag], dim=-1).reshape(
            batch,
            self._num_ue * self._num_subbands,
            2 * 16 * 2,
        )
        noise_feature = torch.log10(noise + 1e-9)
        noise_feature = noise_feature[:, :, None].expand(-1, -1, self._num_subbands)
        noise_feature = noise_feature.reshape(batch, self._num_ue * self._num_subbands, 1)
        values = self._embed(torch.cat([values, noise_feature], dim=-1))
        values = torch.cat([self._prc_token.expand(batch, -1, -1), values], dim=1)
        pooled = self._norm(self._tfm(self._pos(values)))[:, 0]
        values = self._fc_out(pooled).reshape(batch, self._num_subbands, NUM_TX, self._num_ue, 2)
        precoder = torch.complex(values[..., 0], values[..., 1])
        energy = torch.sum(torch.abs(precoder).square(), dim=(2, 3), keepdim=True)
        return precoder / torch.sqrt(energy + 1e-9)


class TransmitterCore(nn.Module):
    def __init__(self):
        super().__init__()
        self._mod = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(NUM_BITS_PER_SYMBOL, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )
                for _ in range(NUM_UE)
            ]
        )
        self._decoder = DecoderCore()
        self._precoder = PrecoderCore()

    @staticmethod
    def _modulate(bits: torch.Tensor, modulator: nn.Module) -> torch.Tensor:
        batch = bits.shape[0]
        values = modulator(bits.reshape(batch, NUM_DL_SC, NUM_BITS_PER_SYMBOL))
        symbols = torch.complex(values[..., 0], values[..., 1])
        energy = torch.mean(torch.abs(symbols).square(), dim=1, keepdim=True)
        return (symbols / torch.sqrt(energy + 1e-9)).unsqueeze(1)

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        symbols = torch.cat(
            [self._modulate(bits_list[user], self._mod[user]) for user in range(NUM_UE)],
            dim=1,
        ).permute(0, 2, 1)
        h_hat = torch.stack([self._decoder(feedback_list[user]) for user in range(NUM_UE)], dim=1)
        noise = torch.pow(10.0, -snr_dl.transpose(0, 1) / 10.0)
        precoder = self._precoder(h_hat, noise).repeat_interleave(48, dim=1)
        signal = torch.matmul(precoder, symbols.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        control = torch.ones(
            (bits_list[0].shape[0], NUM_CTRL),
            device=signal.device,
            dtype=bits_list[0].dtype,
        )
        return signal, control


class ReceiverCore(nn.Module):
    def __init__(self):
        super().__init__()
        width = 128
        input_features = 2 * 2 + 2 * 2 * NUM_TX + 1
        self._embed = nn.Linear(input_features, width)
        self._pos = PositionalEncoding(NUM_DL_SC, width)
        self._tfm = _transformer(width, heads=4, layers=6)
        self._norm = nn.LayerNorm(width)
        self._fc_out = nn.Linear(width, NUM_BITS_PER_SYMBOL)

    def forward(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        del ctrl_bits
        batch = y.shape[0]
        y = y.permute(0, 2, 1)
        y_features = torch.cat([y.real, y.imag], dim=-1)
        h = h.permute(0, 3, 1, 2).reshape(batch, NUM_DL_SC, -1)
        h_features = torch.cat([h.real, h.imag], dim=-1)
        noise_feature = torch.log10(torch.pow(10.0, -snr / 10.0) + 1e-9)[:, None, None]
        noise_feature = noise_feature.expand(batch, NUM_DL_SC, 1)
        values = self._embed(torch.cat([y_features, h_features, noise_feature], dim=-1))
        values = self._norm(self._tfm(self._pos(values)))
        return self._fc_out(values).reshape(batch, NUM_BITS_PER_UE)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([EncoderCore() for _ in range(NUM_EXPERTS)])

    def initialize_from_baseline(self, state_dict: dict[str, torch.Tensor]) -> None:
        for expert in self.experts:
            expert.load_state_dict(state_dict)

    def forward(self, h: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        indices = _expert_indices(snr)
        output = torch.empty((h.shape[0], NUM_UL_RE), device=h.device, dtype=h.dtype)
        for expert_index, expert in enumerate(self.experts):
            selected = indices == expert_index
            if bool(selected.any().item()):
                output[selected] = expert(h[selected], snr[selected])
        return output


class Transmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([TransmitterCore() for _ in range(NUM_EXPERTS)])

    def initialize_from_baseline(self, state_dict: dict[str, torch.Tensor]) -> None:
        for expert in self.experts:
            expert.load_state_dict(state_dict)

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_snr = snr_dl.amin(dim=0)
        indices = _expert_indices(pair_snr)
        batch = bits_list[0].shape[0]
        signal = torch.empty((batch, NUM_TX, NUM_DL_SC), device=feedback_list[0].device, dtype=feedback_list[0].dtype)
        control = torch.empty((batch, NUM_CTRL), device=bits_list[0].device, dtype=bits_list[0].dtype)
        for expert_index, expert in enumerate(self.experts):
            selected = indices == expert_index
            if bool(selected.any().item()):
                expert_signal, expert_control = expert(
                    [bits[selected] for bits in bits_list],
                    [feedback[selected] for feedback in feedback_list],
                    snr_dl[:, selected],
                )
                signal[selected] = expert_signal
                control[selected] = expert_control
        return signal, control


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([ReceiverCore() for _ in range(NUM_EXPERTS)])

    def initialize_from_baseline(self, state_dict: dict[str, torch.Tensor]) -> None:
        for expert in self.experts:
            expert.load_state_dict(state_dict)

    def forward(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        indices = _expert_indices(snr)
        output = torch.empty((y.shape[0], NUM_BITS_PER_UE), device=y.device, dtype=torch.float32)
        for expert_index, expert in enumerate(self.experts):
            selected = indices == expert_index
            if bool(selected.any().item()):
                output[selected] = expert(y[selected], h[selected], ctrl_bits[selected], snr[selected])
        return output

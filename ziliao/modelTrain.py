import os
import torch
import torch.nn as nn
from modelDesign import Encoder, Transmitter, Receiver
import numpy as np


#=======================================================================================================================
#=======================================================================================================================
# System Parameters Setting
NUM_UE = 2

NUM_UPLINK_SUBCARRIERS = 96

NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_CTRL_BITS = 5
NUM_DOWNLINK_TX = 16
NUM_DOWNLINK_RX = 2
NUM_SC_PER_SB = 48

NUM_BITS_PER_RE = 8
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * NUM_BITS_PER_RE

SNR_DL_RANGE = [-20, 20]
SNR_UL_GAP = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#=======================================================================================================================
#=======================================================================================================================
# Link Definition
class MU_MIMO_Link(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder()
        self._transmitter = Transmitter()
        self._receiver = Receiver()

        self._num_ue = NUM_UE
        self._num_uplink_re = NUM_UPLINK_SUBCARRIERS
        self._num_downlink_data_subcarriers = NUM_DOWNLINK_DATA_SUBCARRIERS
        self._num_downlink_ctrl_bits = NUM_DOWNLINK_CTRL_BITS
        self._num_downlink_tx = NUM_DOWNLINK_TX

    def forward(self, h_list, b_list, snr_dl, snr_ul):
        batch_size = h_list[0].shape[0]
        H_list = [torch.as_tensor(h_i, device=DEVICE) for h_i in h_list]
        b_list = [b.to(DEVICE) for b in b_list]
        snr_dl = snr_dl.to(DEVICE)
        snr_ul = snr_ul.to(DEVICE)

        # CSI feedback via uplink channel
        I_list = []
        for i in range(self._num_ue):
            snr_dl_i = snr_dl[i]
            snr_ul_i = snr_ul[i]
            U = self._encoder(H_list[i], snr_dl_i)
            assert U.shape == (batch_size, self._num_uplink_re), "Dimension error!"
            assert U.dtype == torch.complex64, "Data type error!"
            energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
            U = U / torch.sqrt(energy)
            sqrt2 = torch.sqrt(torch.tensor(2.0, device=DEVICE, dtype=torch.float32))
            g = torch.complex(torch.randn_like(U, dtype=torch.float32) / sqrt2, torch.randn_like(U, dtype=torch.float32) / sqrt2)
            n_ul = g * torch.sqrt(torch.reshape(10 ** (-snr_ul_i / 10.0), [-1, 1]))
            I_list.append(U + n_ul)

        # MU transmission
        X, b_ctrl = self._transmitter(b_list, I_list, snr_dl)

        # Dimension check
        assert X.shape == (batch_size, self._num_downlink_tx, self._num_downlink_data_subcarriers), "Dimension error!"
        assert X.dtype == torch.complex64, "Data type error!"
        assert b_ctrl.shape == (batch_size, self._num_downlink_ctrl_bits), "Dimension error!"
        if not torch.all(torch.eq(b_ctrl, b_ctrl * b_ctrl)):
            raise AssertionError("Ctrl bits format error!")

        # Power normalization
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2), keepdim=True)
        X = X / torch.sqrt(energy)

        # MU downlink channel and signal receiving
        c_list = []
        X_exp = X.unsqueeze(1)
        for i in range(self._num_ue):
            snr_dl_i = snr_dl[i]
            Y = torch.sum(H_list[i] * X_exp, dim=2)
            sqrt2 = torch.sqrt(torch.tensor(2.0, device=DEVICE, dtype=torch.float32))
            g = torch.complex(torch.randn_like(Y, dtype=torch.float32) / sqrt2, torch.randn_like(Y, dtype=torch.float32) / sqrt2)
            n_dl = g * torch.sqrt(torch.reshape(10 ** (-snr_dl_i / 10.0), [-1, 1, 1]))
            Y = Y + n_dl
            c_list.append(self._receiver(Y, H_list[i], b_ctrl, snr_dl_i))

        return c_list


#=======================================================================================================================
#=======================================================================================================================
# Data Loading
def _load_channel(path):
    d = np.load(path)
    return d['real'].astype(np.float32) + 1j * d['imag'].astype(np.float32)

H_train = _load_channel('./data_train/H_train.npz')
print(H_train.shape)

#=======================================================================================================================
#=======================================================================================================================
# Training
BATCH_SIZE = 100
NUM_TRAINING_ITERATIONS = 200000
print(f'Using device: {DEVICE}')
mu_link = MU_MIMO_Link().to(DEVICE)
optimizer = torch.optim.Adam(mu_link.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for i in range(NUM_TRAINING_ITERATIONS):
    b_list = [torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32, device=DEVICE) for _ in range(NUM_UE)]
    idx = np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)
    h_batch = H_train[idx]
    h_list = [torch.as_tensor(h_batch[:, i], device=DEVICE) for i in range(NUM_UE)]
    snr_dl = SNR_DL_RANGE[0] + (SNR_DL_RANGE[1] - SNR_DL_RANGE[0]) * torch.rand(NUM_UE, BATCH_SIZE, device=DEVICE)
    snr_ul = snr_dl - SNR_UL_GAP

    llr_list = mu_link(h_list, b_list, snr_dl, snr_ul)
    bce_loss = sum(criterion(llr_list[j], b_list[j]) for j in range(NUM_UE))

    loss = bce_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = sum(((llr_list[j] >= 0).float() == b_list[j].float()).sum() / llr_list[j].numel() for j in range(NUM_UE)) / NUM_UE
    print(f'Iteration {i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss.item():.4f} (BCE {bce_loss.item():.4f}) Acc: {acc.item():.4f}')

os.makedirs('./modelSubmit', exist_ok=True)
torch.save(mu_link._encoder.state_dict(), './modelSubmit/encoder.pth')
torch.save(mu_link._transmitter.state_dict(), './modelSubmit/transmitter.pth')
torch.save(mu_link._receiver.state_dict(), './modelSubmit/receiver.pth')

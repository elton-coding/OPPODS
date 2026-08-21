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
NUM_BITS_PER_RE = 8

NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * NUM_BITS_PER_RE

SNR_DL_RANGE = [-20, 20]
SNR_UL_GAP = 10
FAIRNESS_PERCENTILE = 10
SCORE_ALPHA = 0.7
BATCH_SIZE = 1
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
            assert torch.all(energy > 0), "Uplink signal energy must be positive"
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
        assert torch.all(energy > 0), "Downlink signal energy must be positive"
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
            llr = self._receiver(Y, H_list[i], b_ctrl, snr_dl_i)
            assert llr.ndim == 2 and llr.shape[0] == batch_size
            assert 0 < llr.shape[1] <= NUM_MAX_BITS, f"LLR length must be in (0, {NUM_MAX_BITS}], got {llr.shape[1]}"
            assert llr.dtype == torch.float32
            assert torch.isfinite(llr).all(), "LLR contains NaN/Inf"
            c_list.append(llr)

        return c_list


def compute_score(b_data, llr):
    """Compute per-sample score for each UE in the batch."""
    B_max = b_data.shape[1]
    B = llr.shape[1]
    B_c = ((llr >= 0).float() == b_data[:, :B]).float().sum(dim=1)
    return 100.0 * (B_c + (B_max - B) * 0.5) / B_max

#=======================================================================================================================
#=======================================================================================================================
# Data Loading
def _load_channel(path):
    d = np.load(path)
    return d['real'].astype(np.float32) + 1j * d['imag'].astype(np.float32)

H_train = _load_channel('./data_train/H_train.npz')

#=======================================================================================================================
#=======================================================================================================================
# Model Loading
print(f'Using device: {DEVICE}')
mu_link = MU_MIMO_Link().to(DEVICE)
mu_link._encoder.load_state_dict(torch.load('./modelSubmit/encoder.pth', map_location=DEVICE, weights_only=True))
mu_link._transmitter.load_state_dict(torch.load('./modelSubmit/transmitter.pth', map_location=DEVICE, weights_only=True))
mu_link._receiver.load_state_dict(torch.load('./modelSubmit/receiver.pth', map_location=DEVICE, weights_only=True))
mu_link.eval()

#=======================================================================================================================
#=======================================================================================================================
# Testing
NUM_TESTING_SAMPLES = 2000
ue_scores = []
with torch.no_grad():
    for start in range(0, NUM_TESTING_SAMPLES, BATCH_SIZE):
        end = min(start + BATCH_SIZE, NUM_TESTING_SAMPLES)
        bs = end - start
        #print(f'samples {start}-{end - 1}/{NUM_TESTING_SAMPLES} (batch_size={bs})')
        b_list = [torch.randint(0, 2, (bs, NUM_MAX_BITS), dtype=torch.float32, device=DEVICE) for _ in range(NUM_UE)]
        idx = np.random.choice(H_train.shape[0], bs, replace=False)
        h_sample = H_train[idx]
        h_list = [torch.as_tensor(h_sample[:, i], device=DEVICE) for i in range(NUM_UE)]
        snr_dl = SNR_DL_RANGE[0] + (SNR_DL_RANGE[1] - SNR_DL_RANGE[0]) * torch.rand(NUM_UE, bs, device=DEVICE)
        snr_ul = snr_dl - SNR_UL_GAP

        llr_list = mu_link(h_list, b_list, snr_dl, snr_ul)
        for j in range(NUM_UE):
            ue_scores.extend(compute_score(b_list[j], llr_list[j]).tolist())

ue_scores = np.asarray(ue_scores)
efficiency_score = np.mean(ue_scores)
fairness_score = np.percentile(ue_scores, FAIRNESS_PERCENTILE)
final_score = SCORE_ALPHA * efficiency_score + (1.0 - SCORE_ALPHA) * fairness_score
print('efficiency_score = ' + str(efficiency_score))
print('fairness_score = ' + str(fairness_score))
print('final_score = ' + str(final_score))

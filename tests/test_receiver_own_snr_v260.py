import ast
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_pure_neural_snr_experts import load_model_design

DESIGN = ROOT / "research/pure_neural_v260/modelDesign.py"
module = load_model_design(DESIGN)


def controls(indices):
    values = torch.tensor(indices, dtype=torch.long)
    return ((values[:, None] >> torch.arange(5)) & 1).float()


def test_only_receiver_routing_and_metadata_change_ast():
    original = ast.parse((ROOT / "research/pure_neural_v230/modelDesign.py").read_text())
    candidate = ast.parse(DESIGN.read_text())
    for name in ("EncoderCore", "Encoder", "ResidualMLPBlock", "TransmitterCore", "Transmitter", "ReceiverCore"):
        a = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == name)
        b = next(n for n in candidate.body if isinstance(n, ast.ClassDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b)
    assert module.NUM_BITS_PER_UE == 1152 and module.NUM_EXPERTS == 8
    assert module.TRANSMITTER_ROUTING == "whole_min"


@pytest.mark.parametrize("profile", range(8))
def test_boundaries_and_initial_parent_preserved(profile):
    snr = torch.tensor([-20., -15., -10.001, -10., -5.001, -5., -.001, 0., 4.999, 5., 9.999, 10., 14.999, 15., 20.])
    result = module._receiver_indices(controls([profile] * len(snr)), snr)
    expected = ((snr >= -10).long() if profile < 2 else
                torch.tensor([2,2,2,2,2,3,3,4,4,5,5,6,6,7,7]))
    assert torch.equal(result, expected)
    assert all((r < 2) == (profile < 2) for r in result.tolist())


def test_decode_preserves_all_five_control_bits_and_original_clamping():
    snr = torch.zeros(6)
    assert module._receiver_indices(controls([0,1,2,7,16,31]), snr).tolist() == [1,1,4,4,4,4]


def test_expected_route_probabilities_on_balanced_midpoint_grid():
    x = torch.arange(-19.75, 20., .5)
    own, partner = torch.meshgrid(x, x, indexing="ij")
    own, partner = own.flatten(), partner.flatten()
    tx = module._expert_indices(torch.minimum(own, partner))
    route = module._receiver_indices(controls(tx.tolist()), own)
    frequencies = torch.bincount(route, minlength=8).double() / len(route)
    torch.testing.assert_close(frequencies, torch.tensor([.25,.1875]+[.09375]*6,dtype=torch.float64), atol=0, rtol=0)


class TaggedCore(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, y, h, ctrl, snr):
        return torch.full((len(y),1152), float(self.index), device=y.device)


def test_actual_receiver_forward_uses_new_route_for_batch_and_single():
    receiver = module.Receiver.__new__(module.Receiver)
    nn.Module.__init__(receiver)
    receiver.experts = nn.ModuleList([TaggedCore(i) for i in range(8)])
    snr = torch.tensor([-17.5,20.,-7.5,-2.5,2.5,7.5,12.5,17.5])
    ctrl = controls([0,1,2,3,4,5,6,7])
    y = torch.zeros(8,2,144,dtype=torch.complex64)
    h = torch.zeros(8,2,16,144,dtype=torch.complex64)
    result = receiver(y,h,ctrl,snr)
    assert result.shape == (8,1152)
    torch.testing.assert_close(result[:,0],torch.arange(8).float(),atol=0,rtol=0)
    receiver.eval()
    for i in range(8):
        torch.testing.assert_close(receiver(y[i:i+1],h[i:i+1],ctrl[i:i+1],snr[i:i+1]),result[i:i+1],atol=0,rtol=0)

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_pure_neural_shared_v257 import shared_name
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design


def test_only_low_tx_aliases_change():
    module = load_model_design(ROOT / 'research/pure_neural_v265/modelDesign.py')
    link = PureNeuralLink(module)
    assert module.TX_SHARING_MAP == (0, 1, 2, 2, 2, 2, 2, 2)
    assert module.SHARED_PARENT_MAP == (0, 0, 1, 1, 1, 1, 1, 1)
    for dtype in (torch.float32, torch.float64, torch.float32):
        link.to(dtype=dtype)
        for component in ('transmitter', 'receiver'):
            experts = getattr(link, component).experts
            groups = module.TX_SHARING_MAP if component == 'transmitter' else module.SHARED_PARENT_MAP
            for i in range(8):
                for j in range(i):
                    other = dict(experts[j].named_parameters())
                    for name, parameter in experts[i].named_parameters():
                        expected = groups[i] == groups[j] and shared_name(component, name)
                        assert (parameter is other[name]) == expected
    low = link.transmitter.experts
    link.zero_grad(set_to_none=True)
    (2 * low[0]._blocks[0]._scale + 3 * low[1]._blocks[0]._scale).backward()
    assert low[0]._blocks[0]._scale.grad.item() == 2
    assert low[1]._blocks[0]._scale.grad.item() == 3


def test_source_diff_is_only_sharing_configuration():
    original = (ROOT / 'research/pure_neural_v257/modelDesign.py').read_text(encoding='utf-8')
    candidate = (ROOT / 'research/pure_neural_v265/modelDesign.py').read_text(encoding='utf-8')
    candidate = candidate.replace('\nTX_SHARING_MAP = (0, 1, 2, 2, 2, 2, 2, 2)', '')
    candidate = candidate.replace('def _share_prefix(experts: nn.ModuleList, input_modules: tuple[str, ...],\n                  sharing_map: tuple[int, ...] = SHARED_PARENT_MAP) -> None:', 'def _share_prefix(experts: nn.ModuleList, input_modules: tuple[str, ...]) -> None:')
    candidate = candidate.replace('zip(experts, sharing_map, strict=True)', 'zip(experts, SHARED_PARENT_MAP, strict=True)')
    candidate = candidate.replace('_share_prefix(self.experts, ("_bit_embed", "_feedback_expand", "_embed"), TX_SHARING_MAP)', '_share_prefix(self.experts, ("_bit_embed", "_feedback_expand", "_embed"))')
    assert candidate.strip() == original.strip()

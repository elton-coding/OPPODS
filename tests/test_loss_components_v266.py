import torch
from diagnose_loss_components_v266 import losses


def test_exact_loss_and_gradient_decomposition():
    generator = torch.Generator().manual_seed(26610)
    for zero in (False, True):
        logits = torch.randn(4, 2, 32, generator=generator, dtype=torch.float64)
        if zero:
            logits.zero_()
        logits.requires_grad_(True)
        bits = torch.randint(0, 2, logits.shape, generator=generator).double()
        score, bce, total = losses(logits, bits)
        torch.testing.assert_close(score+bce, total)
        a, b, c = [torch.autograd.grad(v, logits, retain_graph=True)[0] for v in (score, bce, total)]
        assert torch.isfinite(c).all()
        torch.testing.assert_close(a+b, c)

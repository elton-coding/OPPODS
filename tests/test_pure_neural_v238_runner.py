import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("v238_runner", ROOT / "scripts/run_pure_neural_width_v238.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def value(command, flag):
    return command[command.index(flag) + 1]


def test_width_probe_has_real_training_batch_but_separate_output_and_budget():
    experiment = runner.plan()
    formal = runner.train_command(experiment)
    probe = runner.probe_command(experiment)
    assert value(formal, "--learning-rate") == "1e-5"
    assert value(formal, "--steps") == "12000"
    assert value(formal, "--validation-samples") == "2000"
    assert value(probe, "--steps") == "2"
    assert value(probe, "--validation-samples") == "4"
    assert value(probe, "--output-dir") != value(formal, "--output-dir")
    assert experiment["mapping"] == [0, 1]
    for command in (formal, probe):
        assert value(command, "--batch-size") == "100"
        assert value(command, "--gpu-memory-fraction") == "0.4"
        assert value(command, "--baseline-dir") == "artifacts/pure_neural_v227/joint_low"
        assert value(command, "--model-design") == "research/pure_neural_v238/modelDesign.py"

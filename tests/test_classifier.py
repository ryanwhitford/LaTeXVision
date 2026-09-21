import torch

from src.models.classifier import SymbolClassifier


def test_output_shape_matches_num_classes():
    model = SymbolClassifier(num_classes=26)
    x = torch.randn(4, 1, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 26)


def test_forward_is_deterministic_in_eval_mode():
    model = SymbolClassifier(num_classes=10)
    model.eval()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


def test_handles_non_default_input_size():
    # AdaptiveAvgPool2d makes the classifier head agnostic to spatial size.
    model = SymbolClassifier(num_classes=5)
    x = torch.randn(1, 1, 48, 48)
    logits = model(x)
    assert logits.shape == (1, 5)


def test_gradients_flow_through_all_parameters():
    model = SymbolClassifier(num_classes=8)
    x = torch.randn(2, 1, 32, 32)
    target = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"

import torch

from dvoechnik.model import PolicyValueNet
from dvoechnik.trainer import train_episode


def test_self_play_updates_network():
    model = PolicyValueNet()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_episode(model, max_plies=4)
    after = list(model.parameters())
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(old, new) for old, new in zip(before, after))

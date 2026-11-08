import torch

from dvoechnik.model import PolicyValueNet
from dvoechnik.trainer import train_episode
from torch import optim


def test_self_play_updates_network():
    model = PolicyValueNet()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss, result = train_episode(model, optim.Adam(model.parameters(), lr=1e-3), max_plies=4)
    after = list(model.parameters())
    assert torch.isfinite(torch.tensor(loss))
    assert result == "draw"
    assert any(not torch.equal(old, new) for old, new in zip(before, after))

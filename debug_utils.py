__doc__ = """ File for useful debugging functions.  """

import torch


def network_msdiff(net1: torch.nn.Module, net2: torch.nn.Module) -> float:
    """
    Calculate the mean squared difference between two networks
    Args:
        net1:
        net2:

    Returns:

    """

    sd1 = net1.state_dict()
    sd2 = net2.state_dict()
    total_diff = 0
    for key, _ in net1.state_dict().items():
        cur_diff = torch.nn.functional.mse_loss(sd1[key], sd2[key])
        total_diff += cur_diff
    return total_diff.item()

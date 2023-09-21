import torch


def OI_init(model):
    for n, m in model.named_parameters():
        if "bias" in n:
            m.data.fill_(0.0)
        if "weight" in n and m.ndim == 2:
            torch.nn.init.orthogonal_(m.data)


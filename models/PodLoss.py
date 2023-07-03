import torch
from torch.nn import functional as F


class PodLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self,
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        memory_flags=None,
        only_old=False,
        **kwargs
    ):
        """Pooled Output Distillation.
        
        """
        assert len(list_attentions_a) == len(list_attentions_b)

        loss = torch.tensor(0.).to(list_attentions_a[0].device)
        for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
            # shape of (b, n, w, h)
            assert a.shape == b.shape, (a.shape, b.shape)

            if only_old:
                a = a[memory_flags]
                b = b[memory_flags]
                if len(a) == 0:
                    continue

            a = torch.pow(a, 2)
            b = torch.pow(b, 2)

            if collapse_channels == "channels":
                a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
                b = b.sum(dim=1).view(b.shape[0], -1)
            elif collapse_channels == "width":
                a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
                b = b.sum(dim=2).view(b.shape[0], -1)
            elif collapse_channels == "height":
                a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
                b = b.sum(dim=3).view(b.shape[0], -1)
            elif collapse_channels == "gap":
                a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
                b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
            elif collapse_channels == "spatial":
                a_h = a.sum(dim=3).view(a.shape[0], -1)
                b_h = b.sum(dim=3).view(b.shape[0], -1)
                a_w = a.sum(dim=2).view(a.shape[0], -1)
                b_w = b.sum(dim=2).view(b.shape[0], -1)
                a = torch.cat([a_h, a_w], dim=-1)
                b = torch.cat([b_h, b_w], dim=-1)
            else:
                raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

            if normalize:
                a = F.normalize(a, dim=1, p=2)
                b = F.normalize(b, dim=1, p=2)

            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
            loss += layer_loss

        return loss / len(list_attentions_a)
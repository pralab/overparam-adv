import torch
from torch import Tensor
from torch.nn.functional import one_hot


class LipLowerBound:
    r"""Class to estimate a lower bound of the l_2 Lipschitz constant of a model. """

    def __init__(self, model: torch.nn.Module, std:float = 0.125) -> None:
        r"""Initialize model"""
        self.model = model
        self.model.eval()
        self.std = std

    def estimate(self, inputs: Tensor, labels: Tensor, n_aug :int = 100)-> float:
        r"""Esitmate a lower bound of the Lipschitz constant of the model.

        Args:
            inputs: Input to the model.
            labels: Labels corresponding to the input.
            augmentation: number of augmentations to perform for estimation.
        """
        lip_estim = 0.0
        for _ in range(n_aug):
            _inputs = inputs + torch.randn_like(inputs)*self.std
            _inputs = _inputs.clamp(0., 1.)
            _inputs.requires_grad_()
            with torch.enable_grad():
                _outputs = self.model(_inputs)
                for c in range(_outputs.shape[1]):
                    out_diff = _outputs.gather(1, labels.unsqueeze(1)) - _outputs[:,c]
                    grad = torch.autograd.grad(
                        outputs=out_diff.sum(),
                        inputs=_inputs,
                        retain_graph=True
                    )[0]
                    norm_grad = torch.linalg.vector_norm(grad, dim=(1,2,3))
                    lip_estim = max(lip_estim, norm_grad.max().item())
        return lip_estim
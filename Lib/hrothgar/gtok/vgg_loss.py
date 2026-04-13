import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.vgg import VGG19_Weights


class VGG(nn.Module):
    """VGG/Perceptual Loss

    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """

    def __init__(self, conv_index: str = "22"):

        super(VGG, self).__init__()
        vgg_features = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features
        modules = [m for m in vgg_features]

        if conv_index == "22":
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == "54":
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(
        self, model_output: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Compute VGG/Perceptual loss

        Parameters
        ----------
        model_output : torch.Tensor
            Model output tensor
        ground_truth : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between model_output and ground_truth

        """

        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(model_output)

        with torch.no_grad():
            vgg_ground_truth = _forward(ground_truth.detach())

        loss = F.mse_loss(vgg_sr, vgg_ground_truth)

        return loss

import torch
import torch.nn as nn


def orthogonalize(tensor):
    s_norm = torch.linalg.norm(tensor, ord=2)
    W = tensor / s_norm

    I = torch.eye(64, device=tensor.device)
    for _ in range(10):
        W = W @ (1.5 * I - 0.5 * W.T @ W)
    
    return W


class simple_model(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        # 8x8 patch Reconstruction을 위한 간단한 CNN
        # Input: (B, 1, 8, 8) -> Output: (B, 1, 8, 8)
        self.feature = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(4, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.coefficients_predictor = nn.Conv2d(base_channels, 64, 4, 1, 0)

        basis = torch.randn(1, 64, 8, 8)
        self.basis = nn.Parameter(basis, requires_grad=True)

    def forward(self, x):
        feat = self.feature(x)
        coefficients = self.coefficients_predictor(feat)

        basis = orthogonalize(self.basis.view(64, 64)).view(1, 64, 8, 8)
        recon = torch.sum(basis * coefficients, dim=1, keepdim=True)
        return recon

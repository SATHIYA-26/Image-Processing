import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):  # 2 outputs: global (2 classes), mask (1 channel)
        super(UNet, self).__init__()
        self.inc = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.global_conv = nn.Conv2d(64, 2, 1)  # Global classification (2 classes)
        self.mask_conv = nn.Conv2d(64, 1, 1)    # Mask output (1 channel)

    def forward(self, x):
        x1 = nn.functional.relu(self.inc(x))
        x2 = nn.functional.max_pool2d(x1, 2)
        x2 = nn.functional.relu(self.down1(x2))
        x = self.up1(x2)
        global_out = self.global_conv(x)  # [batch, 2, H, W]
        mask_out = self.mask_conv(x)      # [batch, 1, H, W]
        return torch.cat([global_out, mask_out], dim=1)  # [batch, 3, H, W]

if __name__ == "__main__":
    model = UNet()
    print(model)
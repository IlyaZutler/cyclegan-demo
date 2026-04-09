import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_res_blocks: int = 9):
        super().__init__()

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        ch = base_channels
        for _ in range(2):
            encoder += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        transformer = [ResBlock(ch) for _ in range(n_res_blocks)]

        decoder = []
        for _ in range(2):
            decoder += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, in_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *transformer, *decoder)

    def forward(self, x):
        return self.model(x)


def load_generators(weights_path: str, device: torch.device):
    """
    Загружает generators.pt — state_dict от model.generators (nn.ModuleDict).
    Возвращает dict {"a_to_b": Generator, "b_to_a": Generator}.
    """
    generators = nn.ModuleDict({
        "a_to_b": Generator(),
        "b_to_a": Generator(),
    })
    state = torch.load(weights_path, map_location=device)
    generators.load_state_dict(state)
    generators.to(device)
    generators.eval()
    return generators

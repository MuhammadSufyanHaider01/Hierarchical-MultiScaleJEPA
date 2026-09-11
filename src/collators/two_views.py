import torch
import torch.nn as nn
import torchvision.transforms as VT
import torchaudio.transforms as AT


class RandomTimeAxisCrop(nn.Module):
    """Random crop on time axis only, keeps full frequency axis."""

    def __init__(self, output_size: tuple[int, int], scale: tuple[float, float]):
        super().__init__()
        self.output_height, self.output_width = output_size
        self.scale_min, self.scale_max = scale

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        time_axis = spec.dim() - 1
        current_time = spec.shape[time_axis]

        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        target_time = int(current_time * scale)

        if target_time < current_time:
            max_start = current_time - target_time
            start = torch.randint(0, max_start + 1, (1,)).item()
            spec = spec[..., start:start + target_time]

        if spec.shape[time_axis] != self.output_width:
            spec = torch.nn.functional.interpolate(
                spec,
                size=(self.output_height, self.output_width),
                mode='bicubic',
            )

        return spec


class AdditiveGaussianNoise(nn.Module):
    """Additive i.i.d. Gaussian noise."""

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec + torch.randn_like(spec) * self.std


class TwoViewsCollator:
    def __init__(
        self,
        output_size: tuple[int, int],
        crop: bool,
        crop_scale: tuple[float, float],
        n_time_masks: int,
        time_mask_max_width: int,
        n_freq_masks: int,
        freq_mask_max_width: int,
        add_noise: bool,
        noise_std: float,
    ):
        transforms = []
        if crop:
            transforms.append(
                RandomTimeAxisCrop(
                    output_size=output_size,
                    scale=crop_scale,
                )
            )

        for _ in range(n_time_masks):
            transforms.append(
                AT.TimeMasking(time_mask_param=time_mask_max_width)
            )
        for _ in range(n_freq_masks):
            transforms.append(
                AT.FrequencyMasking(freq_mask_param=freq_mask_max_width)
            )

        if add_noise:
            transforms.append(AdditiveGaussianNoise(noise_std))
    
        self.transform = VT.Compose(transforms)

    def __call__(self, batch: list[torch.Tensor]) -> tuple[list[torch.Tensor], ...]:
        collated_batch = torch.utils.data.default_collate(batch)
        view1 = self.transform(collated_batch)
        view2 = self.transform(collated_batch)

        return view1, view2

"""RadImageNet feature extraction.

Wraps DenseNet-121, ResNet-50, and InceptionV3 backbones pretrained on
RadImageNet, converting raw 2-D or 3-D medical images to embedding vectors
suitable for downstream radiogenomic analyses.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)

# ImageNet statistics used by RadImageNet preprocessing
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_EMBED_DIM = {
    "densenet121": 1024,
    "inceptionv3": 2048,
    "resnet50":    2048,
}


def _normalization_likely_performed(image_np: np.ndarray) -> bool:
    """Heuristic: values in a narrow normalised range suggest prior normalisation."""
    arr = image_np.ravel()
    if arr.size == 0:
        return False
    vmin, vmax = float(arr.min()), float(arr.max())
    return vmax <= 5.0 and vmin >= -5.0 and not np.issubdtype(image_np.dtype, np.integer)


class RadImageNetEmbedding(nn.Module):
    """Backbone encoder pretrained on RadImageNet.

    Parameters
    ----------
    base_model_type:
        One of ``"densenet121"`` (default), ``"resnet50"``, or
        ``"inceptionv3"``.  Weights are *not* loaded here; call
        :func:`load_radimagenet_weights` after construction.

    Output shape: ``(B, embed_dim)`` where ``embed_dim`` is 1024 for
    DenseNet-121 and 2048 for the other two architectures.
    """

    def __init__(self, base_model_type: str = "densenet121") -> None:
        super().__init__()
        if base_model_type not in _EMBED_DIM:
            raise ValueError(
                f"Unknown base_model_type {base_model_type!r}. "
                f"Choose one of {list(_EMBED_DIM)}."
            )
        self.base_model_type = base_model_type
        self._embed_dim = _EMBED_DIM[base_model_type]

        if base_model_type == "densenet121":
            base = models.densenet121(weights=None)
            self.backbone = base.features  # (B,1024,H',W') before pooling

        elif base_model_type == "resnet50":
            base = models.resnet50(weights=None)
            # Drop the classification head; keep everything through avgpool
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,2048,1,1)

        else:  # inceptionv3
            base = models.inception_v3(weights=None, aux_logits=False)
            base.fc = nn.Identity()
            self.backbone = base  # returns (B,2048) directly

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, 3, H, W)``

        Returns
        -------
        Tensor of shape ``(B, embed_dim)``
        """
        if self.base_model_type == "densenet121":
            feat = self.backbone(x)                       # (B,1024,H',W')
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))   # (B,1024,1,1)
            return torch.flatten(feat, 1)                 # (B,1024)

        elif self.base_model_type == "resnet50":
            feat = self.backbone(x)                       # (B,2048,1,1)
            return torch.flatten(feat, 1)                 # (B,2048)

        else:  # inceptionv3 — fc=Identity, returns (B,2048)
            return self.backbone(x)


def get_radimagenet_embeddings(
    x: torch.Tensor,
    model: RadImageNetEmbedding,
    fine_tune: bool = False,
) -> torch.Tensor:
    """Run a batch of images through *model* and return embeddings.

    Parameters
    ----------
    x:
        ``(N, 3, 224, 224)`` tensor of preprocessed images.
    model:
        A :class:`RadImageNetEmbedding` instance.
    fine_tune:
        If ``True`` keep gradients attached; otherwise wrap in
        ``torch.no_grad()``.

    Returns
    -------
    ``(N, embed_dim)`` embedding tensor.
    """
    if fine_tune:
        return model(x)
    with torch.no_grad():
        return model(x)


def convert_image_to_radimagenet_format(
    image: "np.ndarray | torch.Tensor",
    target_size: Optional[int] = 224,
) -> torch.Tensor:
    """Preprocess a raw medical image for RadImageNet inference.

    Supports 2-D ``(H, W)`` and 3-D ``(D, H, W)`` inputs.

    * 2-D → ``(3, H, W)``  single pseudo-RGB frame
    * 3-D → ``(D, 3, H, W)``  one pseudo-RGB frame per slice

    Processing steps
    ----------------
    1. Cast to float32.
    2. Min-max normalise to ``[0, 1]``.
    3. Replicate the single channel to produce pseudo-RGB.
    4. Resize spatial dimensions to ``target_size × target_size`` (bilinear).
    5. Apply ImageNet channel normalisation.

    Parameters
    ----------
    image:
        Raw pixel/voxel array.  May be a NumPy array or a PyTorch tensor.
    target_size:
        Target spatial size in pixels.  Pass ``None`` to skip resizing.

    Returns
    -------
    Float32 tensor, shape ``(3, H, W)`` or ``(D, 3, H, W)``.
    """
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"Expected np.ndarray or torch.Tensor, got {type(image).__name__}"
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2-D or 3-D image, got shape {tuple(image.shape)}"
        )

    if torch.is_tensor(image):
        t = image.float()
        arr_np = t.detach().cpu().numpy()
    else:
        arr_np = image.astype(np.float32, copy=False)
        t = torch.from_numpy(arr_np)

    if _normalization_likely_performed(arr_np):
        logger.warning(
            "Image appears to already be normalised. RadImageNet expects raw "
            "pixel values scaled to [0,1] then ImageNet-normalised. "
            "Proceeding anyway — results may be unexpected."
        )

    # Min-max normalise to [0, 1]
    t_min, t_max = t.min(), t.max()
    if torch.isclose(t_max, t_min):
        t = torch.zeros_like(t)
    else:
        t = (t - t_min) / (t_max - t_min)

    device = t.device
    mean = torch.tensor(_IMAGENET_MEAN, device=device)
    std  = torch.tensor(_IMAGENET_STD,  device=device)

    if t.ndim == 2:
        # (H,W) → (1,H,W) → (3,H,W)
        t = t.unsqueeze(0).repeat(3, 1, 1)

        if target_size is not None:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        t = (t - mean[:, None, None]) / std[:, None, None]

    else:
        # (D,H,W) → (D,1,H,W) → (D,3,H,W)
        t = t.unsqueeze(1).repeat(1, 3, 1, 1)

        if target_size is not None:
            t = F.interpolate(
                t,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        t = (t - mean[None, :, None, None]) / std[None, :, None, None]

    return t

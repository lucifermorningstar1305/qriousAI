"""
https://github.com/4m4n5/CLIP-Lite/blob/master/data/transforms.py
"""

import random
from typing import List
import unicodedata
import re

import albumentations as alb
import cv2


class CaptionOnlyTransform(alb.BasicTransform):
    r"""
    A base class for custom `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
    transform, which can transform captions. Captions may be ``str``, or tokens
    (``List[int]``) as per implementation of :meth:`apply_to_caption`. These
    transforms will have consistent API as other transforms from albumentations.
    """

    @property
    def targets(self):
        return {"caption": self.apply_to_caption}

    def apply_to_caption(self, caption, **params):
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        # Super class adds "width" and "height" but we don't have image here.
        return params


class ImageCaptionTransform(alb.BasicTransform):
    r"""
    Similar to :class:`~virtex.data.transforms.CaptionOnlyTransform`, this
    extends super class to work on ``(image, caption)`` pair together.
    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply_to_caption(self):
        raise NotImplementedError


class NormalizeCaption(CaptionOnlyTransform):
    r"""
    Perform common normalization with caption: lowercase, trim leading and
    trailing whitespaces, NFKD normalization and strip accents.
    Examples
    --------
    >>> normalize = NormalizeCaption(always_apply=True)
    >>> out = normalize(caption="Some caption input here.")  # keys: {"caption"}
    """

    def __init__(self, max_caption_length: int = 30):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self.max_caption_length = max_caption_length

    def pre_caption(self, caption: str):
        max_words = self.max_caption_length
        caption = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                caption.lower(),
            )
            .replace("-", " ")
            .replace("/", " ")
            .replace("<person>", "person")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption

    def apply_to_caption(self, caption: str, **params) -> str:
        caption = self.pre_caption(caption)
        caption = caption.lower()
        caption = unicodedata.normalize("NFKD", caption)
        caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])
        return caption


class HorizontalFlip(ImageCaptionTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption.
    .. note::
        This transform can also work on images only (without the captions).
        Its behavior will be same as albumentations
        :class:`~albumentations.augmentations.transforms.HorizontalFlip`.
    Examples
    --------
    >>> flip = HorizontalFlip(p=0.5)
    >>> out1 = flip(image=image, caption=caption)  # keys: {"image", "caption"}
    >>> # Also works with images (without caption).
    >>> out2 = flip(image=image)  # keys: {"image"}
    """

    def apply(self, img, **params):
        return cv2.flip(img, 1)

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        return caption


class RandomResizedSquareCrop(alb.RandomResizedCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.RandomResizedCrop`
    which assumes a square crop (width = height). Everything else is same.
    Parameters
    ----------
    size: int
        Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class CenterSquareCrop(alb.CenterCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.CenterCrop` which
    assumes a square crop (width = height). Everything else is same.
    Parameters
    ----------
    size: int
        Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class SquareResize(alb.Resize):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.Resize` which
    assumes a square resize (width = height). Everything else is same.
    Parameters
    ----------
    size: int
        Dimension of the width and height of the resized image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


# =============================================================================
#   SOME COMMON CONSTANTS AND IMAGE TRANSFORMS:
#   These serve as references here, and are used as default params in many
#   dataset class constructors.
# -----------------------------------------------------------------------------

IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""

IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)
r"""ImageNet color normalization std in RGB format (values in 0-1)."""

DEFAULT_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)

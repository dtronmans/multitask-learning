import random
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import ImageFilter, Image
import torch
import numpy as np

class PairedTransform:
    def __init__(
        self,
        size=(336, 544),
        flip_prob=0.5,
        rotation_degrees=10,
        affine_params=None
    ):
        self.size = size  # (H, W)
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.affine_params = affine_params or {
            "degrees": 10,
            "translate": (0.05, 0.05),
            "scale": (0.95, 1.05),
            "shear": None
        }

    def _apply_same_transform(self, image, mask, transform_func):
        return transform_func(image), transform_func(mask)

    def __call__(self, image, mask):
        # Resize
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size)

        # Horizontal flip
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Rotation
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        if random.random() < self.flip_prob:
            image = F.rotate(image, angle, fill=0)
            mask = F.rotate(mask, angle, fill=0)

        # Affine
        if random.random() < 0.2:
            affine_angle = random.uniform(-self.affine_params["degrees"], self.affine_params["degrees"])
            translate = (
                int(self.affine_params["translate"][0] * self.size[1]),
                int(self.affine_params["translate"][1] * self.size[0])
            )
            scale = random.uniform(*self.affine_params["scale"])
            shear = self.affine_params["shear"] or [0.0, 0.0]

            image = F.affine(image, affine_angle, translate, scale, shear, fill=0)
            mask = F.affine(mask, affine_angle, translate, scale, shear, fill=0)

        # Gaussian blur
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius))

        # Perspective transform
        if random.random() < 0.2:
            perspective = T.RandomPerspective(distortion_scale=0.2, p=1.0)
            image = perspective(image)
            mask = perspective(mask)

        # Brightness/contrast adjustment (mild)
        if random.random() < 0.2:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)

        # Random noise addition (grayscale)
        if random.random() < 0.2:
            np_img = np.array(image).astype(np.float32)
            noise = np.random.normal(0, 5, np_img.shape)  # small noise
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(np_img)

        # Convert to tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        return image, mask

class DefaultPairedTransform:
    def __init__(self, size=(336, 544)):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size)

        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        return image, mask
import random
import torchvision.transforms.functional as F
from PIL import Image

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

    def __call__(self, image, mask):
        # Resize first
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random rotation
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        # image = F.rotate(image, angle, fill=0)
        # mask = F.rotate(mask, angle, fill=0)

        # Random affine
        affine_angle = random.uniform(-self.affine_params["degrees"], self.affine_params["degrees"])
        translate = (
            int(self.affine_params["translate"][0] * self.size[1]),  # width
            int(self.affine_params["translate"][1] * self.size[0])   # height
        )
        scale = random.uniform(*self.affine_params["scale"])
        shear = self.affine_params["shear"] or [0.0, 0.0]

        # image = F.affine(image, affine_angle, translate, scale, shear, fill=0)
        # mask = F.affine(mask, affine_angle, translate, scale, shear, fill=0)

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
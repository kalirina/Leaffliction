from torchvision.transforms import v2
from PIL import Image
import os
import json
import shutil
import random


def main():
    if not os.path.exists("data/stats.json"):
        print("Run distribution.py first")
        return
    with open("data/stats.json","r") as file:
        stats = json.load(file)
    max_distrib = max(stats.values())

    data_dir = "data/leaves/images"
    aug_dir = "data/augmented_directory"
    transforms = {
        "Flip":v2.RandomHorizontalFlip(p=1.0),
        "Rotate":v2.RandomRotation(degrees=45),
        "Skew":v2.RandomPerspective(distortion_scale=0.2, p=1.0),
        "Shear": v2.RandomAffine(degrees=0, shear=45),
        "Crop": v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 0.8)),
        "Distortion":v2.ElasticTransform(alpha=50.0)
    }

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        aug_class_path = os.path.join(aug_dir, class_dir)
        os.makedirs(aug_class_path, exist_ok=True)
        files = os.listdir(class_path)
        for file in files:
            shutil.copy2(os.path.join(class_path, file),os.path.join(aug_class_path, file))
        miss_distrib = max_distrib - stats[class_dir]
        generated = 0
        for file in os.listdir(class_path):
            if generated >= miss_distrib:
                break
            img = Image.open(os.path.join(class_path, file)).convert("RGB")
            name, _ = os.path.splitext(file)
            for t_name, transform in transforms.items():
                if generated >= miss_distrib:
                    break
                out_img = transform(img)
                out_path = os.path.join(aug_class_path,f"{name}_{t_name}.JPG")
                out_img.save(out_path)
                generated += 1


if __name__ == '__main__':
    main()

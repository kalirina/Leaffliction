from torchvision.transforms import v2
from PIL import Image
import os


def main():
    data_dir = "data/leaves/images"
    augmented_dir = "data/augmented_directory"
    transforms = {
        "Flip":v2.RandomHorizontalFlip(p=1.0),
        "Rotate":v2.RandomRotation(degrees=45),
        "Skew":v2.RandomPerspective(distortion_scale=0.2, p=1.0),
        "Shear": v2.RandomAffine(degrees=0, shear=45),
        "Crop": v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 0.8)),
        "Distortion":v2.ElasticTransform(alpha=50.0)
    }
    ex_p = "data/leaves/images/Apple_Black_rot/image (1).JPG"
    img = Image.open(ex_p).convert("RGB")
    os.makedirs(augmented_dir, exist_ok=True)
    for t_name, transform in transforms.items():
        out_img = transform(img)
        out_img.save("data/augmented_directory/image (1)_" + t_name + ".JPG")
    # for class_dir in os.listdir(data_dir):
    #     for file in os.listdir("data/leaves/images/" + dir):
    #         img = Image.open(img_path).convert("RGB")

if __name__ == '__main__':
    main()

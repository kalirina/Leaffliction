import sys
import os
import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import cv2
import numpy as np


def generate_histogram(img, mask):
    """Recreates the 9-channel color histogram on a single plot."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    total_pixels = cv2.countNonZero(mask)
    assert total_pixels > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Color Histogram", fontsize=14, fontweight='bold')

    rgb_channels = ['Red', 'Green', 'Blue']
    hsv_channels = ['Hue', 'Saturation', 'Value']
    lab_channels = ['Lightness', 'Green-Magenta', 'Blue-Yellow']
    color_data = [
        (rgb_img, rgb_channels, ['red', 'green', 'blue']),
        (hsv_img, hsv_channels, ['purple', 'cyan', 'gray']),
        (lab_img, lab_channels, ['black', 'magenta', 'gold'])
    ]

    for img_space, channel_names, colors in color_data:
        for i, (name, color) in enumerate(zip(channel_names, colors)):
            hist = cv2.calcHist([img_space], [i], mask, [256], [0, 256])
            hist_percent = (hist / total_pixels) * 100
            ax.plot(hist_percent, color=color, label=name)

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity", fontsize=12)
    ax.set_ylabel("Proportion of Pixels (%)", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    return fig


def display_transformations(images_dict, dest_dir, filename):
    """Plots a dictionary of images in a grid"""
    if dest_dir:
        # Save each image as a separate file
        for title, img in images_dict.items():
            if title == "Original":
                continue
            # Clean up the dictionary key to use as a valid filename suffix
            suffix = title.replace(" ", "_").lower()
            out_path = os.path.join(dest_dir, f"{filename}_{suffix}.png")
            cv2.imwrite(out_path, img)
        print(f"Saved transformations to: {out_path}")

    else:
        # Display grid
        num_images = len(images_dict)
        cols = 3
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, (title, img) in enumerate(images_dict.items()):
            ax = axes[i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=14)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def transform_image(img_path, dest_dir=None):
    """Applies PlantCV transformations to a single image."""
    pcv.params.debug = None

    if dest_dir:
        print(f"Processing and saving: {img_path}")
    else:
        print(f"Displaying transformations for: {img_path}")

    img, path, filename = pcv.readimage(filename=img_path)
    height, width, _ = img.shape

    # Basic image process process pipeline
    # We blur the image, then extract b_channel from lab format and build
    # the mask from that (white where b value its high)
    blurred_img = pcv.gaussian_blur(img=img, ksize=(11, 11),
                                    sigma_x=0, sigma_y=None)
    b_channel = pcv.rgb2gray_lab(rgb_img=blurred_img, channel='b')
    raw_mask = pcv.threshold.otsu(gray_img=b_channel, object_type='light')
    clean_mask = pcv.fill(bin_img=raw_mask, size=50)
    clean_mask = pcv.fill_holes(bin_img=clean_mask)

    # ROI Object filtering
    # We select our Region Of Interest by placing a square centered on the
    # image anything that is not inside the square gets filtered out
    roi_x, roi_y = width//6, height//6
    roi_w, roi_h = (width//6)*4, (height//6)*4
    roi = pcv.roi.rectangle(img=img, x=roi_x, y=roi_y, h=roi_h, w=roi_w)
    filtered_mask = pcv.roi.filter(mask=clean_mask, roi=roi,
                                   roi_type='partial')

    # Shape Analysis
    # First we label (separate) the different objects we have in the mask,
    # based on wether the pixels touch eachother or not, then we pass the
    # label and mask to .size which calculates area, perimeter, center
    # of mass etc and returns the image with these data draw
    labeled_mask, num_objects = pcv.create_labels(mask=filtered_mask)
    shape_image = pcv.analyze.size(img=img, labeled_mask=labeled_mask)

    # Quantitative Color Analysis
    # Extract median, mean and standard deviation for RGB, HSV and LAB
    color_histogram_img = pcv.analyze.color(rgb_img=img, labeled_mask=filtered_mask, colorspaces="all")

    # Images preparation before displaying
    masked_img = img.copy()
    # Paint bg pixels white
    masked_img[clean_mask == 0] = [255, 255, 255]

    roi_objects_img = img.copy()
    # Paint plant pixels green
    roi_objects_img[filtered_mask > 0] = [0, 255, 0]
    cv2.rectangle(roi_objects_img, (roi_x, roi_y), (roi_x + roi_w,
                  roi_y + roi_h), (255, 0, 0), 4)  # Blue box

    # Pseudolandmarks
    left, right, center = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=filtered_mask
        )
    landmark_img = img.copy()

    point_styles = [
        ((255, 0, 0), left),     # BGR Blue
        ((255, 0, 255), right),  # BGR Magenta
        ((0, 165, 255), center)  # BGR Orange
    ]
    for color, points in point_styles:
        for pt in points:
            x, y = np.array(pt).flatten()[:2]
            cv2.circle(landmark_img, (int(x), int(y)), radius=5,
                       color=color, thickness=-1)

    # Histogram
    hist_fig = generate_histogram(img=img, mask=filtered_mask)

    images_to_display = {
        "Original": img,
        "Mask": clean_mask,
        "Masked image": masked_img,
        "ROI filtering": roi_objects_img,
        "Shape Analysis": shape_image,
        "Pseudolandmarks": landmark_img
    }

    base_name = os.path.splitext(filename)[0]
    if dest_dir:
        hist_path = os.path.join(dest_dir, f"{base_name}_histogram.png")
        hist_fig.savefig(hist_path)
        plt.close(hist_fig)

        csv_path = os.path.join(dest_dir, f"{base_name}_measurements.csv")
        pcv.outputs.save_results(filename=csv_path, outformat="csv")
        print(f"Saved numerical measurements to: {csv_path}")

    display_transformations(images_to_display, dest_dir, base_name)

    pcv.outputs.clear()

def main():
    parser = argparse.ArgumentParser(prog="TransformImage",
                                     description="Transform images")
    parser.add_argument("src", type=str, help="Source directory or file path")
    parser.add_argument("-dst", type=str, default=None,
                        help="Path to save the output")

    args = parser.parse_args()
    valid_exts = ('.png', '.jpg', '.jpeg')

    if not os.path.exists(args.src):
        print(f"Error: Source path '{args.src}' does not exist")
        sys.exit(1)

    if os.path.isfile(args.src):
        if not args.src.lower().endswith(valid_exts):
            print(f"Error: '{args.src}' is not a valid file extension")
            sys.exit(1)
        transform_image(args.src, dest_dir=None)

    elif os.path.isdir(args.src):
        if not args.dst:
            print("Error: You must provide a destination directory (-dst).")
            sys.exit(1)
        os.makedirs(args.dst, exist_ok=True)
        for file in os.listdir(args.src):
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(args.src, file)
                transform_image(full_path, args.dst)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

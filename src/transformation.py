import sys
import os
import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import cv2
import numpy as np

def generate_9_channel_histogram(img, mask):
    """Recreates the 9-channel color histogram on a single plot."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    total_pixels = cv2.countNonZero(mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("9-Channel Color Histogram", fontsize=14, fontweight='bold')

    color_data = [
        (img, ['Red', 'Green', 'Blue'], ['red', 'green', 'blue']),
        (hsv_img, ['Hue', 'Saturation', 'Value'], ['purple', 'cyan', 'gray']),
        (lab_img, ['Lightness', 'Green-Magenta', 'Blue-Yellow'], ['black', 'magenta', 'gold'])
    ]

    for img_space, channel_names, colors in color_data:
        for i, (name, color) in enumerate(zip(channel_names, colors)):
            hist = cv2.calcHist([img_space], [i], mask, [256], [0, 256])
            hist_percent = (hist / total_pixels) * 100 if total_pixels > 0 else hist
            ax.plot(hist_percent, color=color, label=name)

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity / Bin (0-255)", fontsize=12)
    ax.set_ylabel("Proportion of Pixels (%)", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    return fig

def display_with_titles(images_dict, dest_dir, filename):
    """Plots a dictionary of images in a neat grid with titles."""
    num_images = len(images_dict)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, (title, img) in enumerate(images_dict.items()):
        ax = axes[i]
        if img is not None:
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

        ax.set_title(title, fontsize=14)
        ax.axis('off')

    for j in range(len(images_dict), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if dest_dir:
        out_path = os.path.join(dest_dir, f"{filename}_transformations.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved transformation grid to: {out_path}")
    else:
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

    # 1. Processing pipeline
    blurred_img = pcv.gaussian_blur(img=img, ksize=(11, 11), sigma_x=0, sigma_y=None)
    b_channel = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    # This raw mask represents Figure IV.2 in your screenshot
    raw_mask = pcv.threshold.binary(gray_img=b_channel, threshold=115, object_type='light')
    clean_mask = pcv.fill(bin_img=raw_mask, size=200)

    # 2. ROI Object filtering
    roi_x, roi_y, roi_w, roi_h = width//4, height//4, width//2, height//2
    roi = pcv.roi.rectangle(img=img, x=roi_x, y=roi_y, h=roi_h, w=roi_w)
    filtered_mask = pcv.roi.filter(mask=clean_mask, roi=roi, roi_type='partial')

    # 3. Shape Analysis (Figure IV.5)
    labeled_mask, num_objects = pcv.create_labels(mask=filtered_mask)
    shape_image = pcv.analyze.size(img=img, labeled_mask=labeled_mask)

    # --- CUSTOM VISUALIZATIONS FOR ASSIGNMENT --- #

    # Figure IV.3: Mask (Original image with white background)
    masked_img = img.copy()
    masked_img[clean_mask == 0] = [255, 255, 255] # Turn background pixels white

    # Figure IV.4: Roi objects (Blue box, green mask overlay)
    roi_objects_img = img.copy()
    roi_objects_img[filtered_mask > 0] = [0, 255, 0] # Paint plant pixels green (RGB)
    cv2.rectangle(roi_objects_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 4) # Blue box

    # Figure IV.6: Pseudolandmarks (Draw points matching assignment colors)
    top, bottom, center = pcv.homology.y_axis_pseudolandmarks(img=img, mask=filtered_mask)
    landmark_img = img.copy()

    # Colors (RGB): Top=Blue, Bottom=Magenta, Center=Orange
    point_styles = [
        ((0, 0, 255), top),      # Blue
        ((255, 0, 255), bottom), # Magenta
        ((255, 165, 0), center)  # Orange
    ]
    for color, points in point_styles:
        for pt in points:
            x, y = np.array(pt).flatten()[:2]
            cv2.circle(landmark_img, (int(x), int(y)), radius=5, color=color, thickness=-1)

    # 4. Generate Histogram
    hist_fig = generate_9_channel_histogram(img=img, mask=filtered_mask)

    # --- DICTIONARY MATCHING SCREENSHOT --- #
    images_to_display = {
        "Figure IV.1: Original": img,
        "Figure IV.2: Gaussian blur": raw_mask,
        "Figure IV.3: Mask": masked_img,
        "Figure IV.4: Roi objects": roi_objects_img,
        "Figure IV.5: Analyze object": shape_image,
        "Figure IV.6: Pseudolandmarks": landmark_img
    }

    base_name = os.path.splitext(filename)[0]
    if dest_dir:
        hist_path = os.path.join(dest_dir, f"{base_name}_histogram.png")
        hist_fig.savefig(hist_path)
        plt.close(hist_fig)

    display_with_titles(images_to_display, dest_dir, base_name)


def main():
    parser = argparse.ArgumentParser(prog="TransformImage", description="Transform images")
    parser.add_argument("src", type=str, help="Source directory or file path")
    parser.add_argument("-dst", type=str, default=None, help="Path to save the output")

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

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Reducing the number of intensity levels in an image."""
def reduce_intensity_levels(image, levels):
    factor = 255 // (levels - 1)
    reduced_image = (image // factor) * factor
    return reduced_image.astype(np.uint8)

"""Applies a mean filter with given kernel size."""
def average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

"""Rotates image by given angle in degrees."""
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h))

"""Reduces spatial resolution by block averaging."""
def block_average(image, block_size):
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            avg = np.mean(block, axis=(0, 1), keepdims=True)
            out[y:y+block_size, x:x+block_size] = avg
    return out.astype(np.uint8)

"""Displays an image using matplotlib"""
def display_image(title, img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Load grayscale and color image
    img_path = './tulips.png' 
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(img_path)

    if gray is None or color is None:
        print("Error: Image not found.")
        return

    ### 1. Reduce intensity levels
    for level in [2, 4, 8, 16, 32]:
        reduced = reduce_intensity_levels(gray, level)
        cv2.imwrite(f'output/reduced_{level}.png', reduced)

    ### 2. Spatial averaging (smoothing)
    for k in [3, 10, 20]:
        blurred = average_filter(gray, k)
        cv2.imwrite(f'output/blurred_{k}x{k}.png', blurred)

    ### 3. Rotate image
    for angle in [45, 90]:
        rotated = rotate_image(color, angle)
        cv2.imwrite(f'output/rotated_{angle}.png', rotated)

    ### 4. Block averaging for resolution reduction
    for block in [3, 5, 7]:
        reduced_spatial = block_average(color, block)
        cv2.imwrite(f'output/block_avg_{block}x{block}.png', reduced_spatial)

if __name__ == "__main__":
    main()

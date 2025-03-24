import cv2
import numpy as np

def remove_watermark(image_path, output_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read the image. Check the file path.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to detect watermark
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the mask to cover more area
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Apply inpainting
    result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Watermark removed! Image saved at {output_path}")

# Paths
image_path = "C:/Users/Varun Raghu/Pictures/Watermark2.jpg"
output_path = "C:/Users/Varun Raghu/Pictures/Watermark_removed4.jpg"

remove_watermark(image_path, output_path)

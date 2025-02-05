import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for contrast stretching (linear contrast)
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    contrast_image = (image - min_val) * (255 / (max_val - min_val))
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image

# Function for logarithmic transformation
def logarithmic_transformation(image):
    image = image.astype(np.float32)        # Convert to float for precision
    c = 255 / np.log(1 + np.max(image))     # Scaling constant based on max pixel value
    log_image = c * np.log(1 + image)       # Apply the log transformation
    log_image = np.clip(log_image, 0, 255)  # Ensure values are in the 0-255 range
    return log_image.astype(np.uint8)       # Convert back to uint8

# Function for gamma correction
def gamma_correction(image, gamma = 1.0):
    gamma_image = np.power(image / 255.0, gamma) * 255
    gamma_image = np.clip(gamma_image, 0, 255).astype(np.uint8)
    return gamma_image

# Load an image
image = cv2.imread('./images/images1.jpeg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

if image is None:
    print("Error: Image not found!")
    exit()

# Apply transformations
contrast_image = contrast_stretching(image)
log_image = logarithmic_transformation(image)
gamma_image = gamma_correction(image, 2.0)          # Example gamma value

# Display the results
plt.figure(figsize=(12, 12))

# Original image
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Histogram of original image
plt.subplot(3, 3, 2)
plt.hist(image.ravel(), bins = 256, range = (0, 255), color = 'blue', alpha = 0.7)
plt.title('Histogram: Original Image')

# Logarithmic transformation image
plt.subplot(3, 3, 3)
plt.imshow(log_image, cmap='gray')
plt.title('Logarithmic Transformation')
plt.axis('off')

# Histogram of logarithmic transformation image
plt.subplot(3, 3, 4)
plt.hist(log_image.ravel(), bins = 256, range = (0, 255), color = 'green', alpha = 0.7)
plt.title('Histogram: Log Transformation')

# Contrast stretching image
plt.subplot(3, 3, 5)
plt.imshow(contrast_image, cmap='gray')
plt.title('Contrast Stretching')
plt.axis('off')

# Gamma correction image
plt.subplot(3, 3, 6)
plt.imshow(gamma_image, cmap='gray')
plt.title('Gamma Correction')
plt.axis('off')

# Histogram of contrast stretching image
plt.subplot(3, 3, 7)
plt.hist(contrast_image.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
plt.title('Histogram: Contrast Stretching')

# Histogram of gamma correction image
plt.subplot(3, 3, 8)
plt.hist(gamma_image.ravel(), bins=256, range=(0, 255), color='orange', alpha=0.7)
plt.title('Histogram: Gamma Correction')

# Show the images
plt.tight_layout()
plt.show()
import cv2
import matplotlib.pyplot as plt

def show_images_and_histograms(images, titles, cmap="gray"):
    """
    Display images and their histograms in a single figure.

    Parameters:
    - images: List of images (numpy arrays)
    - titles: List of titles for each image and histogram
    - cmap: Colormap to use for images
    """
    n = len(images)
    plt.figure(figsize=(15, 10))
    
    for i in range(n):
        # Display image
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
        
        # Display histogram
        plt.subplot(2, n, n + i + 1)
        plt.hist(images[i].ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        plt.title(f"Histogram of {titles[i]}")
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def histogram_equalization_demo(image):
    """Perform histogram equalization on an image."""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    return gray_image, equalized_image

# Load the uploaded image
uploaded_image_path = "./images/image.jpg"
image = cv2.imread(uploaded_image_path)

if image is None:
    print("Error: Unable to load the image. Please check the file path or upload a valid image.")
else:
    # Perform histogram equalization
    gray_image, equalized_image = histogram_equalization_demo(image)

    # Show all images and histograms in one graph
    show_images_and_histograms(
        [gray_image, equalized_image], 
        ["Original Grayscale Image", "Histogram Equalized Image"], 
        cmap="gray"
    )

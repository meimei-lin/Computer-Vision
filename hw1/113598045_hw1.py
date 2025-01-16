import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Grayscale conversion function
def convertGray(img):
    height, width, channels = img.shape
    gray_img = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            # Get the RGB value of each pixel
            r_value = img[i, j, 2]
            g_value = img[i, j, 1]
            b_value = img[i, j, 0]
            # Calculate the gray value
            gray_value = int(0.3 * r_value + 0.59 * g_value + 0.11 * b_value)
            gray_img[i, j] = gray_value

    return gray_img

# Read a RGB image
img = cv2.imread('test_img/CKS.jpg')
# Convert the image to grayscale image
gray_img = convertGray(img)
# Save as CKS _Q1.jpg
cv2.imwrite('result_img/CKS_Q1.jpg', gray_img)

def ZeroPadding(nparray):
    # Create a padded_array,the size is 2 rows and 2 columns larger than the original array
    padded_array = np.zeros((nparray.shape[0]+2, nparray.shape[1]+2))
    for i in range(nparray.shape[0]): # the number of rows (i.e. height) of the array
        for j in range(nparray.shape[1]): # the number of columns (i.e. width) of the array
            # Each element in the original array is copied to the corresponding position in the new (padded_array)
            padded_array[i+1][j+1] = nparray[i][j]
    plt.imshow(padded_array, cmap='gray')
    plt.title("Padded image of shape:" + str(padded_array.shape))
    plt.show()
    return  padded_array

kernal = [0, -1,  0,
          -1,  4 -1,
           0, -1, 0]

def edge_detection(img, row, col):
    pixels = []
    for i in range(row, row + 3):
        for j in range(col, col + 3):
            pixels.append(img[i, j])
    # Multiply corresponding sections of pixel matrix and convolution kernel
    value = sum([x * y for x, y in zip(pixels, kernal)])
    return value

def ConvolutionOperation(gray_img):
    # Convert grayscale image to int32 to prevent overflow
    gray_img = gray_img.astype(np.int32)
    # Apply zero padding to the grayscale image
    padded_img = ZeroPadding(gray_img)
    # Get the dimensions of the padded image
    padded_height, padded_width = padded_img.shape

    edge_detect_image = []

    # Apply edge detection (convolution) on the padded image using a 3x3 kernel
    # Now we loop through the padded image (ignoring the padding boundary)
    for row in range(padded_height - 2):  # Avoid index out of bounds
        image_row = []
        for col in range(padded_width - 2):
            value = edge_detection(padded_img, row, col)
            image_row.append(value)
        edge_detect_image.append(image_row)

    # Convert the list to a NumPy array for further processing
    arr = np.array(edge_detect_image)

    # Normalize the output to fit into 0-255 range (important for image display)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return  arr

edge_detected_img = ConvolutionOperation(gray_img)
# Convert NumPy array to image and save as CKS_Q2.jpg
image_out = Image.fromarray(edge_detected_img)
image_out.save('result_img/CKS_Q2.jpg')


def Pooling(img, pool_size=3, stride=2):
    # Get the dimensions of the input image
    height, width = img.shape[:2] # range:0~1

    # Calculate the dimensions of the output image based on the stride
    pooling_height = (height - pool_size) // stride + 1
    pooling_width = (width - pool_size) // stride + 1

    # Create an empty array for the pooled image
    max_pool_img = np.zeros((pooling_height, pooling_width), dtype=np.uint8)
    average_pool_img = np.zeros((pooling_height, pooling_width), dtype=np.uint8)

    # Iterate over the input image and apply max pooling and average pooling
    for i in range(pooling_height):
        for j in range(pooling_width):
            # Define the region for pooling with the stride
            region = img[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            # Maximum value in the region
            max_pool_img[i, j] = np.max(region)
            # Average value in the region
            average_pool_img[i, j] = np.mean(region)
    return max_pool_img,average_pool_img
output2_img = cv2.imread('result_img/CKS_Q2.jpg')
max_pool_img,average_pool_img = Pooling(output2_img)
cv2.imwrite('result_img/CKS_Q3a.jpg', average_pool_img)
cv2.imwrite('result_img/CKS_Q3b.jpg', max_pool_img)

def Binarization(img, threshold):
    img = convertGray(img)
    height, width = img.shape
    binar_img = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i, j] < threshold:
                binar_img[i, j] = 0
            else:
                binar_img[i, j] = 255

    return binar_img

output_q3a = cv2.imread('result_img/CKS_Q3a.jpg')
output_q3b = cv2.imread('result_img/CKS_Q3b.jpg')
binar_averagePool_img = Binarization(output_q3a,50)
binar_maxPool_img = Binarization(output_q3b,50)
cv2.imwrite('result_img/CKS_Q4a.jpg',binar_averagePool_img)
cv2.imwrite('result_img/CKS_Q4b.jpg',binar_maxPool_img)




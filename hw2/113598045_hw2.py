import numpy as np
import cv2
import math
from math import pi,exp
import matplotlib.pyplot as plt
import pandas as pd

def calculateMSE(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    height, width = img1.shape
    # Calculate the squared difference of each pixel
    squared_diff_sum = 0
    for i in range(height):
        for j in range(width):
            diff = float(img1[i, j]) - float(img2[i, j]) # Convert to floating point number for calculation
            squared_diff_sum += diff ** 2
    # Total pixels
    total_pixels = height * width
    # Calculate MSE
    mse = squared_diff_sum / total_pixels
    return mse

def calculatePSNR(original_img, filtered_img):
    # Calculate MSE
    mse = calculateMSE(original_img, filtered_img)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0 # Maximum pixel value
    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    psnr = round(psnr, 2)
    return psnr

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def padding(img, pad_size):
    height, width = img.shape
    # Create a new image with padding (larger size)
    padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=img.dtype)
    # Copy the original image into the center of the padded image
    padded_image[pad_size:height + pad_size, pad_size:width + pad_size] = img.copy()

    # Fill the top and bottom padding with edge values
    padded_image[:pad_size, pad_size:width + pad_size] = img[0:1, :]  # Top edge
    padded_image[height + pad_size:, pad_size:width + pad_size] = img[-1:, :]  # Bottom edge

    # Fill the left and right padding with edge values
    padded_image[pad_size:height + pad_size, :pad_size] = img[:, :1]  # Left edge
    padded_image[pad_size:height + pad_size, width + pad_size:] = img[:, -1:]  # Right edge

    # Fill the corners
    padded_image[:pad_size, :pad_size] = img[0, 0]  # Top-left corner
    padded_image[:pad_size, width + pad_size:] = img[0, -1]  # Top-right corner
    padded_image[height + pad_size:, :pad_size] = img[-1, 0]  # Bottom-left corner
    padded_image[height + pad_size:, width + pad_size:] = img[-1, -1]  # Bottom-right corner
    return padded_image

def Median_filter(img, kernel_size, stride):
    height, width = img.shape
    # Get the padding size (half of the kernel size)
    pad_size = kernel_size // 2
    # Pad the image using the padding function
    padded_image = padding(img, pad_size)
    # Create an empty array for the filtered image
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    # Iterate over every pixel in the image using the stride
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            # Extract the neighborhood (kernel) around the current pixel from the padded image
            window = padded_image[i:i + kernel_size, j:j + kernel_size].flatten()
            # Sort the window using the quicksort function
            sorted_window = quicksort(window.tolist())
            # Calculate the median and assign it to the filtered image
            median_value = sorted_window[len(sorted_window) // 2]
            filtered_image[i // stride, j // stride] = median_value
    return filtered_image

def Gaussian_kernel(kernel_size, sigma):
    # Calculate center point
    center = kernel_size // 2
    # Create an empty array
    kernel = np.zeros((kernel_size, kernel_size))
    # Calculate the value of the Gaussian kernel
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            # Gaussian kernel equation
            kernel[i, j] = (1 / (2 * pi * sigma ** 2)) * exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]
    # normalization
    kernel = kernel / sum_val
    return kernel

def Gaussian_filter(img, kernel_size, stride):
    height, width = img.shape
    # Get the padding size (half of the kernel size)
    pad_size = kernel_size // 2
    # Pad the image using the padding function
    padded_image = padding(img, pad_size)
    # Create an empty array for the filtered image
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    kernel = Gaussian_kernel(kernel_size, 1)
    # Iterate over every pixel in the image using the stride
    for i in range(0, height , stride):
        for j in range(0, width , stride):
            # Extract the neighborhood (kernel) around the current pixel from the padded image
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            value = (window * kernel).sum() # Accumulate the values of window * kernel
            filtered_image[i // stride, j // stride] = value

    return filtered_image

# Apply Median and Gaussian filters in combination
def combine_median_and_gaussian(noisy_img, median_kernel_size, gaussian_kernel_size, stride):
    # Apply Median Filter
    median_filtered_img = Median_filter(noisy_img, median_kernel_size, stride)
    # Apply Gaussian Filter
    final_filtered_img = Gaussian_filter(median_filtered_img, gaussian_kernel_size, stride)
    return final_filtered_img

def count_pixel_intensity(img):
    # Count occurrences of each pixel value (0-255)
    pixel_counts = np.zeros(256, dtype=int)
    for i in range(256):
        pixel_counts[i] = np.sum(img == i)
    return pixel_counts

# Function to plot and save histogram
def plot_histogram(img, title, output_file, table_title):
    # Get pixel intensity counts
    pixel_counts = count_pixel_intensity(img)
    pixel_values = np.arange(256)  # Pixel values (0-255)

    # Check if the sum of pixel counts equals total pixels in the image
    total_pixels = img.shape[0] * img.shape[1]
    assert np.sum(pixel_counts) == total_pixels, "The sum of pixel counts must equal the number of pixels in the image"
    # Create a DataFrame for the pixel intensity counts
    print("----------------------------------------------------")
    print(table_title)
    # Display the table
    df = pd.DataFrame({'Pixel Value': pixel_values, 'Count': pixel_counts})
    print(df)

    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Number")
    plt.bar(pixel_values, pixel_counts, width=1)  # Plot a bar chart
    plt.xlim([0, 255])  # Set the x-axis limit to 0-255
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    # Read a noisy and gray images
    img = cv2.imread('test_img/noisy_image.png', cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread('test_img/gray_image.png', cv2.IMREAD_GRAYSCALE)
    # median filter
    median_filter_img = Median_filter(img, kernel_size=3, stride=1)
    # Save as output_q1.png
    cv2.imwrite('result_img/output_q1.png', median_filter_img)
    # Compare PSNR
    psnr = calculatePSNR(img_gray, median_filter_img)
    print("PSNR value of the original image and the image after median filtering:", psnr, "dB")

    # Gaussian filter
    Gaussian_filter_img = Gaussian_filter(img, kernel_size=5, stride=1)
    cv2.imwrite('result_img/output_q2.png', Gaussian_filter_img)
    # Compare PSNR
    psnr = calculatePSNR(img_gray, Gaussian_filter_img)
    print("PSNR value of the original image and the image after Gaussian filtering:", psnr, "dB")

    # Combined filters
    filtered_img = combine_median_and_gaussian(img, median_kernel_size=3, gaussian_kernel_size=3, stride=1)
    cv2.imwrite('result_img/output_q3.png', filtered_img)
    psnr = calculatePSNR(img_gray, filtered_img)
    print("PSNR value after applying combined filters:", psnr, "dB")

    # Plot histograms for each image
    plot_histogram(img, "Histogram of Noisy Image", "result_img/img_noise_his.png", "Noisy Image Statistics")
    plot_histogram(median_filter_img, "Histogram of Median Filtered Image", "result_img/output_q1_his.png", "Median Filtered Image Statistics")
    plot_histogram(Gaussian_filter_img, "Histogram of Gaussian Filtered Image", "result_img/output_q2_his.png", "Gaussian Filtered Image Statistics")
    plot_histogram(filtered_img, "Histogram of Combined Filtered Image", "result_img/output_q3_his.png", "Combined Filtered Image Statistics")
    #plot_histogram(img_gray, "Histogram of Gray Image", "result_img/gray_his.png", "Gray Image Statistics")
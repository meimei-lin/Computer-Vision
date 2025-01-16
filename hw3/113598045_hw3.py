import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi,exp

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

def Canny_edge_detection(img, low_threshold, high_threshold):
    # Step 1: Gradient calculation
    # Apply Sobel operator to calculate gradients in x and y directions
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient magnitude and angle
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    angle = np.arctan2(Gy, Gx) * (180 / np.pi)  # Convert to degrees
    angle = angle % 180  # Map angle to 0-180 degrees range
    # Step 2: Non-maximum suppression
    nms = np.zeros_like(magnitude, dtype=np.float32)
    # Execute non-maximum suppression for each pixel (Exclude boundary pixels to avoid crossing boundaries)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            # Determine the direction of the edge based on the angle
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180): # 0° direction (right and left)
                neighbor1, neighbor2 = magnitude[i, j + 1], magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5: # 45° direction (bottom-left and top-right)
                neighbor1, neighbor2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:  # 90° direction (bottom and top)
                neighbor1, neighbor2 = magnitude[i + 1, j], magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:  # 135° direction (top-left and bottom-right)
                neighbor1, neighbor2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            else:
                neighbor1, neighbor2 = magnitude[i - 1, j], magnitude[i + 1, j]
            # Suppress non-maximum values
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0
    # Step 3: Double threshold and edge tracking by hysteresis
    strong_pixel = 255
    weak_pixel = 50
    result = np.zeros_like(nms, dtype=np.uint8)
    # Strong edges
    strong_i, strong_j = np.where(nms >= high_threshold) # Find all pixels that exceed high_threshold
    result[strong_i, strong_j] = strong_pixel
    # Weak edges
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold)) # Find pixels between low_threshold and high_threshold
    result[weak_i, weak_j] = weak_pixel
    # Edge tracking by hysteresis
    # Iterate over each weak edge pixel of the entire image (excluding borders)
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if result[i, j] == weak_pixel:
                # Check if the weak pixel is connected to a strong pixel
                if np.any(result[i - 1 : i + 1, j - 1 : j + 1] == strong_pixel):
                    result[i, j] = strong_pixel
                else:
                    result[i, j] = 0
    return result

def hough_transform(edge_img, theta_res, rho_res, theta_min, theta_max, mask_top=False):
    height, width = edge_img.shape
    if mask_top :
        # Limit the processing area to the bottom half of the image
        edge_img[:height // 2, :] = 0  # Mask out the upper half of the image
    diag_len = int(np.sqrt(height**2 + width**2))  # Maximum possible rho value
    rhos = np.arange(-diag_len, diag_len, rho_res)  # Range of rho values
    thetas = np.deg2rad(np.arange(theta_min, theta_max, theta_res))  # Range of theta values (converted to radians)

    # Initialize the counter matrix to 0
    counter = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    # Iterate over edge pixel in the image
    edge_pixel = np.argwhere(edge_img)  # Get all non-zero pixel coordinates
    for y, x in edge_pixel:
        for t_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta)) # Calculate rho
            rho_idx = np.argmin(np.abs(rhos - rho))  # Find the index of closest matching ρ value from the discrete array
            counter[rho_idx, t_idx] += 1  # Increment the counter at those theta and rho value
    return rhos, thetas, counter

def draw_lines(img, counter, rhos, thetas, threshold):
    output = img.copy()
    for rho_idx, theta_idx in zip(*np.where(counter > threshold)):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        # Convert polar coordinates (rho, theta) to Cartesian coordinates for the line
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = a * rho, b * rho
        # Calculate two points to draw the line
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        # Draw the line on the image
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return output

if __name__ == '__main__':
    # Read three images
    img1 = cv2.imread('test_img/img1.png')
    img2 = cv2.imread('test_img/img2.png')
    img3 = cv2.imread('test_img/img3.png')
    # Convert the images to grayscale image
    gray_img1 = convertGray(img1)
    gray_img2 = convertGray(img2)
    gray_img3 = convertGray(img3)
    # Gaussian Blur/Filter
    gaussian_img1 = Gaussian_filter(gray_img1, 5, 1)
    gaussian_img2 = Gaussian_filter(gray_img2, 5, 1)
    gaussian_img3 = Gaussian_filter(gray_img3, 5, 1)
    # Save the Gaussian images
    cv2.imwrite('result_img/img1_q1.png', gaussian_img1)
    cv2.imwrite('result_img/img2_q1.png', gaussian_img2)
    cv2.imwrite('result_img/img3_q1.png', gaussian_img3)
    # Canny Edge Detection
    canny_img1 = Canny_edge_detection(gaussian_img1, 30, 180)
    canny_img2 = Canny_edge_detection(gaussian_img2, 30, 80)
    canny_img3 = Canny_edge_detection(gaussian_img3, 30, 180)

    # canny_img1 processing
    height1, width1 = canny_img1.shape
    # Enhanced Canny detection of the bottom half of the image
    bottom_half = canny_img1[height1 // 2:, :]
    local_edges = Canny_edge_detection(bottom_half, 20, 100)
    # Merge the enhanced results back into the original image
    canny_img1[height1 // 2:, :] = local_edges

    # canny_img2 processing
    height2, width2 = canny_img2.shape
    # Define bottom-right corner region
    right_region = canny_img2[height2 // 5:, width2 // 5:]
    # Perform enhanced Canny edge detection on right region
    enhanced_right = Canny_edge_detection(right_region, 20, 90)
    # Merge the enhanced results back into the original image
    canny_img2[height2 // 5:, width2 // 5:] = enhanced_right

    # canny_img3 processing
    height3, width3 = canny_img3.shape
    # Mask out the first 1/4 of the canny image
    canny_img3[:height3 // 4, :] = 0
    # Save the Canny edge detection images
    cv2.imwrite('result_img/img1_q2.png', canny_img1)
    cv2.imwrite('result_img/img2_q2.png', canny_img2)
    cv2.imwrite('result_img/img3_q2.png', canny_img3)
    # Hough Transform
    img1_rhos, img1_thetas, img1_counter = hough_transform(canny_img1, theta_res=4, rho_res=1, theta_min=30, theta_max=180, mask_top=True)
    img2_rhos, img2_thetas, img2_counter = hough_transform(canny_img2, theta_res=4, rho_res=1, theta_min=-90, theta_max=90, mask_top=True)
    img3_rhos, img3_thetas, img3_counter = hough_transform(canny_img3, theta_res=2, rho_res=1, theta_min=-90, theta_max=150, mask_top=False)
    # Draw lines
    output_img1 = draw_lines(img1, img1_counter, img1_rhos, img1_thetas, 95)
    output_img2 = draw_lines(img2, img2_counter, img2_rhos, img2_thetas, 70)
    output_img3 = draw_lines(img3, img3_counter, img3_rhos, img3_thetas, 90)
    # Save the result images
    cv2.imwrite('result_img/img1_q3.png', output_img1)
    cv2.imwrite('result_img/img2_q3.png', output_img2)
    cv2.imwrite('result_img/img3_q3.png', output_img3)
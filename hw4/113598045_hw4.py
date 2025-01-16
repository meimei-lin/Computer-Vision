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

def preprocess_image(img):
    # Convert to grayscale image
    gray_img = convertGray(img)
    # Gaussian filter
    gaussian_img = Gaussian_filter(gray_img, 5, 1)
    # Calculate the gradient of image by using Sobel Operator (Gx and Gy)
    Gx = cv2.Sobel(gaussian_img, cv2.CV_64F, 1, 0, ksize=5)
    Gy = cv2.Sobel(gaussian_img, cv2.CV_64F, 0, 1, ksize=5)
    # Calculate gradient magnitude
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = np.uint8((magnitude / np.max(magnitude)) * 255)
    return gray_img, gaussian_img, magnitude

# Set Initial Contour:ellipse
def initialize_contour_ellipse(img, ellipse_scale_x, ellipse_scale_y, num_points=70, center=None):
    # Get the image dimensions
    height, width = img.shape[:2]
    # Determine the center of the ellipse
    if center is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center
    # Calculate the ellipse dimensions
    a = int(width * ellipse_scale_x) # horizontal
    b = int(height * ellipse_scale_y) # vertical
    # Generate the contour points
    initial_contour = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        x = int(center_x + a * np.cos(t))
        y = int(center_y + b * np.sin(t))
        initial_contour.append([x,y])
    return np.array(initial_contour)

# Set Initial Contour:rectangle
def initialize_contour_rec(img, rect_scale_x, rect_scale_y, num_points=70, center=None):
    # Get the image dimensions
    height, width = img.shape[:2]
    # Determine the center of the rectangle
    if center is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center
    # Calculate side length of rectangle
    rect_width = int(width * rect_scale_x)
    rect_height = int(height * rect_scale_y)
    # Count the number of points on the four sides of a rectangle
    points_per_side = num_points // 4
    initial_contour = []
    # Top edge
    for i in range(points_per_side):
        x = center_x - rect_width // 2 + i * (rect_width // points_per_side)
        y = center_y - rect_height // 2
        initial_contour.append([x, y])
    # Right edge
    for i in range(points_per_side):
        x = center_x + rect_width // 2
        y = center_y - rect_height // 2 + i * (rect_height // points_per_side)
        initial_contour.append([x, y])
    # Bottom edge
    for i in range(points_per_side):
        x = center_x + rect_width // 2 - i * (rect_width // points_per_side)
        y = center_y + rect_height // 2
        initial_contour.append([x, y])
    # Left edge
    for i in range(points_per_side):
        x = center_x - rect_width // 2
        y = center_y + rect_height // 2 - i * (rect_height // points_per_side)
        initial_contour.append([x, y])
    return np.array(initial_contour)

def calculate_energy_components(prev_point, curr_point, next_point, magnitude):
    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    # Continuity energy (Econt)
    Econt = euclidean_distance(curr_point, prev_point)
    # Curvature energy (Ecurv)
    if next_point is not None:
        # pi+1 (next_point) = 2pi (curr_point) - pi-1 (prev_point)
        mid_point = [2 * curr_point[0] - prev_point[0], 2 * curr_point[1] - prev_point[1]]
        Ecurv = euclidean_distance(next_point, mid_point)
    else:
        Ecurv = 0
    # Image energy (Eimg)
    x, y = curr_point
    # Ensure that the coordinates (x,y) of the current point do not exceed the boundaries of the image
    if 0 <= x < magnitude.shape[1] and 0 <= y < magnitude.shape[0]:
        Eimg = -float(magnitude[int(y), int(x)])
    else:
        Eimg = float('inf')
    return Econt, Ecurv, Eimg

def active_contour_step(points, magnitude, alpha, beta, gamma, threshold, min_range_x, max_range_x, min_range_y, max_range_y):
    # Store the updated positions of contour points after the iteration.
    new_points = []
    for i, point in enumerate(points):
        prev_point = points[i - 1]
        next_point = points[(i + 1) % len(points)] if i + 1 < len(points) else None
        Emin = float('inf')
        best_point = point
        for dx in range(min_range_x, max_range_x):
            for dy in range(min_range_y, max_range_y):
                candidate = point + np.array([dx, dy])
                # Calculate energy of candidate points
                Econt, Ecurv, Eimg = calculate_energy_components(prev_point, candidate, next_point, magnitude)
                Etotal = alpha * Econt + beta * Ecurv + gamma * Eimg
                if Etotal < Emin:
                    Emin = Etotal # Update Emin
                    best_point = candidate
                    if Emin - Etotal < threshold:
                        break
        new_points.append(best_point)
    return np.array(new_points)

def active_contour(img, initial_contour, magnitude, max_iterations, convergence_threshold, alpha, beta, gamma, threshold,
                   min_range_x, max_range_x, min_range_y, max_range_y):
    # Active Contour Implementation
    points = initial_contour.copy()
    energy_change = float('inf')
    iteration = 0
    contours_video = []
    while iteration < max_iterations and energy_change > convergence_threshold:
        new_points = active_contour_step(points, magnitude, alpha, beta, gamma, threshold, min_range_x, max_range_x, min_range_y, max_range_y)

        energy_change = np.sum([((new_points[i][0] - points[i][0])**2 + (new_points[i][1] - points[i][1])**2)**0.5 for i in range(len(points))]) / len(points)
        points = new_points
        # Draw updated contour on the image
        frame = img.copy()
        for i in range(len(points)):
            cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), (0, 0, 255), 2)
        contours_video.append(frame)
        if energy_change < convergence_threshold:
            break
        iteration += 1
    return points, contours_video, iteration

def calculate_dsc(ground_truth_mask, predicted_segmentation_mask):
    # Calculate DSC
    intersection = np.sum((ground_truth_mask / 255) * (predicted_segmentation_mask / 255))
    union = np.sum((ground_truth_mask / 255)) + np.sum((predicted_segmentation_mask / 255))
    return 2 * intersection / union

def save_video(video_frames, output_path):
    # Save results as video
    height, width, layers = video_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in video_frames:
        out.write(frame)
    out.release()

def save_image_with_contour(image, contour_points, output_path, point_color ,line_color, point_radius=5):
    output_image = image.copy()
    for i in range(len(contour_points)):
        cv2.line(output_image, tuple(contour_points[i - 1]), tuple(contour_points[i]), line_color, 2)
    for point in contour_points:
        cv2.circle(output_image, tuple(point), radius=point_radius, color=point_color,thickness=-1)
    cv2.imwrite(output_path, output_image)

if __name__ == '__main__':
    # Read three images
    img1 = cv2.imread('test_img/img1.png')
    img2 = cv2.imread('test_img/img2.png')
    img3 = cv2.imread('test_img/img3.png')

    gray_img1, gaussian_img1, magnitude_img1 = preprocess_image(img1)
    cv2.imwrite('result_img/img1_q1.png', magnitude_img1)
    gray_img2, gaussian_img2, magnitude_img2 = preprocess_image(img2)
    cv2.imwrite('result_img/img2_q1.png', magnitude_img2)
    gray_img3, gaussian_img3, magnitude_img3 = preprocess_image(img3)
    cv2.imwrite('result_img/img3_q1.png', magnitude_img3)
    # Initialize contour
    initial_contour_img1 = initialize_contour_ellipse(img1, ellipse_scale_x=0.3, ellipse_scale_y=0.45)
    save_image_with_contour(img1, initial_contour_img1, 'result_img/img1_q2.png', point_color=(0,0,0), line_color=(0,0,0))
    initial_contour_img2 = initialize_contour_ellipse(img2, ellipse_scale_x=0.4, ellipse_scale_y=0.25)
    save_image_with_contour(img2, initial_contour_img2, 'result_img/img2_q2.png', point_color=(0, 0, 0), line_color=(0, 0, 0))
    initial_contour_img3 = initialize_contour_rec(img3, rect_scale_x=0.9, rect_scale_y=0.85)
    save_image_with_contour(img3, initial_contour_img3, 'result_img/img3_q2.png', point_color=(0, 0, 0),line_color=(0, 0, 0))
    # Run active contour algorithm
    final_contour_img1, contours_video_img1, iterations_img1 = active_contour(img1, initial_contour_img1, magnitude_img1, max_iterations=200, convergence_threshold=0.85,
                                                                              alpha=1.5, beta=1.5, gamma=2.0, threshold=1.0,min_range_x=-15, max_range_x=2, min_range_y=-1, max_range_y=17)
    save_image_with_contour(img1, final_contour_img1, 'result_img/img1_q3.png', point_color=(0,0,255), line_color=(0,0,255))
    final_contour_img2, contours_video_img2, iterations_img2 = active_contour(img2, initial_contour_img2,magnitude_img2, max_iterations=200,convergence_threshold=0.85,
                                                                              alpha=1.5,beta=1.5, gamma=2.0, threshold=1.0,min_range_x=-4, max_range_x=4,min_range_y=-1, max_range_y=5)
    save_image_with_contour(img2, final_contour_img2, 'result_img/img2_q3.png', point_color=(0, 0, 255),line_color=(0, 0, 255))
    final_contour_img3, contours_video_img3, iterations_img3 = active_contour(img3, initial_contour_img3,magnitude_img3, max_iterations=200,convergence_threshold=0.75, alpha=1.0,
                                                                              beta=1.0, gamma=3.0, threshold=1.0,min_range_x=-18, max_range_x=4,min_range_y=-1, max_range_y=14)
    save_image_with_contour(img3, final_contour_img3, 'result_img/img3_q3.png', point_color=(0, 0, 255), line_color=(0, 0, 255))
    # Create ground truth (for DSC calculation)
    ground_truth_mask_img1 = cv2.threshold(gray_img1, 127, 255, cv2.THRESH_BINARY)[1]
    ground_truth_mask_img2 = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_BINARY)[1]
    ground_truth_mask_img3 = cv2.threshold(gray_img3, 127, 255, cv2.THRESH_BINARY)[1]
    # Create the predicted segmentation mask
    active_contour_mask1 = np.zeros_like(gray_img1)
    cv2.fillPoly(active_contour_mask1, [final_contour_img1.astype(np.int32)], 255)
    active_contour_mask2 = np.zeros_like(gray_img2)
    cv2.fillPoly(active_contour_mask2, [final_contour_img2.astype(np.int32)], 255)
    active_contour_mask3 = np.zeros_like(gray_img3)
    cv2.fillPoly(active_contour_mask3, [final_contour_img3.astype(np.int32)], 255)
    # Calculate DSC
    dsc_img1 = calculate_dsc(ground_truth_mask_img1, active_contour_mask1)
    dsc_img2 = calculate_dsc(ground_truth_mask_img2, active_contour_mask2)
    dsc_img3 = calculate_dsc(ground_truth_mask_img3, active_contour_mask3)
    # Save video
    save_video(contours_video_img1, "result_img/img1_q4.mp4")
    save_video(contours_video_img2, "result_img/img2_q4.mp4")
    save_video(contours_video_img3, "result_img/img3_q4.mp4")
    print("-------------img1----------------------------")
    print(f"Iterations_img1: {iterations_img1}")
    print(f"DSC_img1: {dsc_img1:.4f}")
    print("-------------img2----------------------------")
    print(f"Iterations_img2: {iterations_img2}")
    print(f"DSC_img2: {dsc_img2:.4f}")
    print("-------------img3----------------------------")
    print(f"Iterations_img3: {iterations_img3}")
    print(f"DSC_img3: {dsc_img3:.4f}")




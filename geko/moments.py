#probably need to delete this model! or check what I need it for in the run_mock


import cv2
import numpy as np
import matplotlib.pyplot as plt
import run_mock as rm

def create_high_res_mask(shape, center, radius, angle_start, angle_end, scale_factor):
    # Create a high-resolution mask
    high_res_shape = (shape[0] * scale_factor, shape[1] * scale_factor)
    high_res_mask = np.zeros(high_res_shape, dtype=np.uint8)
    
    # Define the high-resolution wedge region
    high_res_center = (center[0], center[1])
    high_res_axes = (radius, radius)
    angle = 0
    startAngle = angle_start
    endAngle = angle_end
    thickness = -1  # Filled wedge
    
    # Draw the wedge on the high-resolution mask
    cv2.ellipse(high_res_mask, high_res_center, high_res_axes, angle, startAngle, endAngle, (255,), thickness)
    
    # plt.imshow(high_res_mask, cmap='gray')
    # plt.title("High-Resolution Mask (Wedge)")
    # plt.show()

    high_res_mask = high_res_mask.astype(np.float32)/255

    # # Downsample the high-resolution mask to the original image resolution
    # mask = cv2.resize(high_res_mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    # mask = mask.astype(np.float32) / 255  # Normalize to range [0, 1]
    
    return high_res_mask

def calculate_flux(image, center, radius, angle_start, angle_end, scale_factor=10):
    # Create the high-resolution mask and downsample to the original image size
    mask = create_high_res_mask(image.shape, center, radius, angle_start, angle_end, scale_factor)
    
    #create high resolution image
    high_res_image = cv2.resize(np.array(image), (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
    high_res_image/=(scale_factor**2)

    # sobel_image_y = np.abs(cv2.Sobel(np.array(high_res_image), cv2.CV_64F, 0, 1, ksize=3))
    # sobel_image_x = np.abs(cv2.Sobel(np.array(high_res_image), cv2.CV_64F, 1, 0, ksize=3))
    # sobel_image = np.sqrt(sobel_image_x**2 + sobel_image_y**2)

    # plt.imshow(sobel_image, origin = 'lower')
    # plt.title('Sobel Image')
    # plt.show()
    #plot high res image
    # plt.imshow(high_res_image, cmap='gray')
    # plt.title("High-Resolution Image")
    # plt.show()
    # Apply the mask to the image
    masked_image = high_res_image * mask
    
    # # Calculate the flux (sum of pixel intensities) in the wedge region
    # flux = np.sum(masked_image)
    #Calculate the flux in the wedge region but weigh the pixels by their distance from the center
    flux = 0
    for i in range(masked_image.shape[0]):
        for j in range(masked_image.shape[1]):
            if mask[i,j]>0:
                flux+=masked_image[i,j]*np.sqrt((center[0]-i)**2+(center[1]-j)**2)*2
    # flux = np.sum(masked_image)
    return flux, masked_image

def calculate_std(image, center, radius, angle_start, angle_end, scale_factor=10):
    # Create the high-resolution mask and downsample to the original image size
    mask = create_high_res_mask(image.shape, center, radius, angle_start, angle_end, scale_factor)
    
    #create high resolution image
    high_res_image = cv2.resize(np.array(image), (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_AREA)
    high_res_image/=(scale_factor**2)

    sobel_image_y = np.abs(cv2.Sobel(np.array(image), cv2.CV_64F, 0, 1, ksize=3))
    sobel_image_x = np.abs(cv2.Sobel(np.array(image), cv2.CV_64F, 1, 0, ksize=3))
    sobel_image = np.sqrt(sobel_image_x**2 + sobel_image_y**2)


    #plot high res image
    # plt.imshow(high_res_image, cmap='gray')
    # plt.title("High-Resolution Image")
    # plt.show()
    # Apply the mask to the image
    masked_image = sobel_image * mask
    
    # # Calculate the flux (sum of pixel intensities) in the wedge region
    # flux = np.sum(masked_image)
    #Calculate the flux in the wedge region but weigh the pixels by their distance from the center
    std = np.std(masked_image)
    
    return std, masked_image

def find_max_angle(image, center, radius, increment=5, scale_factor=10):
    max_flux = 1000
    max_angle = 0
    best_masked_image = None
    for i in range(0, 180 + increment, increment):
        print(i)
        angle_start = i
        angle_end = i+increment
        # Calculate the flux and get the masked image
        flux, masked_image = calculate_flux(image, center, radius, angle_start, angle_end, scale_factor=scale_factor)
        if flux<max_flux:
            max_flux = flux
            max_angle = angle_start+increment/2
            best_masked_image = masked_image
        if i == 90:
            print(flux)
    return max_flux, max_angle

def find_min_std_angle(image, center, radius, increment=5, scale_factor=10):
    min_std = 1000
    max_angle = 0
    for i in range(0, 360, increment):
        angle_start = i
        angle_end = i + increment
        # Calculate the flux and get the masked image
        std, masked_image = calculate_std(image, center, radius, angle_start, angle_end, scale_factor=scale_factor)
        if std<min_std:
            min_std = std
            max_angle = angle_start+increment/2
    return min_std, max_angle

def main():
    # Load the image (convert to grayscale for simplicity)
    # image_path = 'path_to_your_image.jpg'
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = rm.make_image(90,60,1,50,None,32)

    plt.imshow(image, origin = 'lower')
    plt.title('Image')
    plt.show()

    sobel_image_y = np.abs(cv2.Sobel(np.array(image), cv2.CV_64F, 0, 1, ksize=3))
    sobel_image_x = np.abs(cv2.Sobel(np.array(image), cv2.CV_64F, 1, 0, ksize=3))
    sobel_image = np.sqrt(sobel_image_x**2 + sobel_image_y**2)

    # # sobel_image = np.abs(cv2.spatialGradient(np.array(image), cv2.CV_64F,ksize=3))

    plt.imshow(sobel_image, origin = 'lower')
    plt.title('Sobel Image')
    plt.show()

    scale_factor=10
    increment=2
    
    # Define the center, radius, and angles of the wedge
    r_t = 1
    r_eff = (1.676/0.4)*r_t
    center = (image.shape[1]*scale_factor // 2 + scale_factor//2, image.shape[0]*scale_factor // 2 + scale_factor//2)
    radius = round(r_eff//4)*scale_factor  # Ensure the radius fits within the image

    max_flux, max_angle = find_max_angle(sobel_image, center, radius, increment=increment, scale_factor=scale_factor)
    print(f"Max flux: {max_flux} at angle {max_angle} ({90-(360-max_angle)%90})")

    # min_std, max_angle = find_min_std_angle(image, center, radius, increment=increment, scale_factor=scale_factor)
    #project the angle 0 -360 angle to 0 - 90
    # print(f"Min std: {min_std} at angle {max_angle%90}")

    #plot the wedge at angle with max flux
    angle_start = max_angle - increment/2
    angle_end = max_angle + increment/2
    flux, masked_image = calculate_flux(sobel_image, center, radius, angle_start, angle_end, scale_factor=scale_factor)
    # plt.imshow(masked_image, cmap='gray')
    # plt.title("Masked Image (Wedge)")
    # plt.show()

    # # Print the calculated flux
    # print(f"Calculated flux in the wedge: {flux}")
    
    # Display the original and masked images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Sobel Image")
    plt.imshow(sobel_image, cmap='gray', origin = 'lower')
    
    plt.subplot(1, 2, 2)
    plt.title("Masked Image (Wedge)")
    plt.imshow(masked_image, cmap='gray', origin = 'lower')
    
    plt.show()

if __name__ == "__main__":
    main()
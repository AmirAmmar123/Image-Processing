import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import math

def present_image(img, title, xlabel = '', ylabel = ''):
  # convert each of the RGB chnnels to the correct dynamic range:
  img = (((img-np.min(img))/np.max(img))* 255).astype(int);
  # Plot:
  plt.figure(figsize=(16,8));
  plt.imshow(img, cmap="gray");
  plt.title(title);
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show();

def bilinear_interpolation_shift(image, dx, dy):
    if dx < 0 or dy < 0 : return image
    # Get input image shape
    height, width = image.shape[:2]
    
    if dx >= 1:
      dx = dx/width 
    if dy >= 1:
      dy = dy/height
    # Generate grid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate new coordinates based on displacement
    new_x = grid_x + dx * width
    new_y = grid_y + dy * height

    # Perform bilinear interpolation
    new_image = map_coordinates(image, [new_y, new_x], order=1, mode='constant')

    return new_image.reshape(image.shape)




# img = cv2.imread('./imgs/cameraman.jpg',cv2.IMREAD_GRAYSCALE)
# plt.subplot(1,2,1)
# plt.title('Original')
# plt.imshow(img, cmap="gray")

# dx, dy = 20, 10
# new_image = bilinear_interpolation_shift(img, dx, dy)
# plt.subplot(1,2,2)
# plt.title(f'bilinear_interpolation_shift with dx={dx}, dy={dy}')
# plt.imshow(new_image, cmap="gray")
# plt.show()





# Load the image
img = cv2.imread('./imgs/Brad.jpg', cv2.IMREAD_GRAYSCALE)

# Create a mask of zeros with the same shape as the image
mask1 = np.zeros_like(img)

# Get the height and width of the image
height, width = mask1.shape

# Define the center and radius of the semicircle
center = (width // 2, height)
radius = width // 2

# Create a semicircular mask
for y in range(height):
    for x in range(width):
        if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
            mask1[y, x] = 1

# Display the mask
# plt.imshow(mask1, cmap='gray')
# plt.title('Mask 1')
# plt.show()

new_img = np.zeros(img.shape)

for x in range(mask1.shape[0]):
  for y in range(mask1.shape[1]):
    if mask1[x, y] == 1:
      new_img[x,y] = img[x,y]
    else:
      new_img[x,y] = 0

# plt.imshow(new_img, cmap='gray')
# plt.title('brad_win')
# plt.show()



def rotate_image(image, angle):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate center of the image
    center_x = width / 2
    center_y = height / 2
    
    # Create an empty rotated image
    rotated_image = np.zeros_like(image)
    
    # Iterate over all pixels in the rotated image
    for y in range(height):
        for x in range(width):
            # Calculate new coordinates after rotation
            new_x = int((x - center_x) * math.cos(angle_rad) - (y - center_y) * math.sin(angle_rad) + center_x)
            new_y = int((x - center_x) * math.sin(angle_rad) + (y - center_y) * math.cos(angle_rad) + center_y)
            
            # Check if the new coordinates are within the bounds of the original image
            if 0 <= new_x < width and 0 <= new_y < height:
                # Assign pixel value from original image to rotated image
                rotated_image[new_y, new_x] = image[y, x]
    
    return rotated_image
  
  
  
# Load the image
image = cv2.imread('./imgs/Brad.jpg', cv2.IMREAD_GRAYSCALE)

# Rotate the image by angles 60°, 45°, and -90°
angles = [60, 45, -90]
rotated_images = [rotate_image(image, angle) for angle in angles]

# Display the results
plt.figure(figsize=(12, 4))
for i in range(len(angles)):
    plt.subplot(1, len(angles), i+1)
    plt.imshow(rotated_images[i], cmap='gray')
    plt.title(f'Angle: {angles[i]}°')
    plt.axis('off')
plt.show()
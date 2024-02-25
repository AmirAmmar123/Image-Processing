import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

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
  


laplcian = np.array([
                  [0,1,0],
                  [1,-4,1],
                  [0,1,0]
                  ])

delta = np.array([ 
                  [0,0,0],
                  [0,1,0],
                  [0,0,0]
                  ])

def convolve_clip(Img, kernel_edge_sharpen, min_val, max_val):
  convolved_image = convolve2d(Img, kernel_edge_sharpen, mode='same')  # Perform 2D convolution
  # Clip the values
  clipped_data = np.clip(convolved_image, min_val, max_val)
  return clipped_data


def salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Add salt and pepper noise to an image.
    
    Parameters:
        image: numpy.ndarray
            Input image.
        salt_prob: float
            Probability of adding salt noise (range: 0 to 1).
        pepper_prob: float
            Probability of adding pepper noise (range: 0 to 1).
    
    Returns:
        numpy.ndarray
            Image with salt and pepper noise.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < salt_prob
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image

I1 = cv2.imread('./imgs/Inigo.jpg', cv2.IMREAD_GRAYSCALE)

a = [x for x in range(1,11)] 

# Define the clipping range
min_val = 0
max_val = 255


# for index, value in enumerate(a):
#   kernel_edge_sharpen = delta - value * laplcian
  # plt.subplot(1, 2, 1)
  # plt.title("orginal")
  # plt.imshow(I1, cmap="gray")
  # plt.subplot(1, 2, 2)
  # plt.imshow(convolve_clip(I1,kernel_edge_sharpen,min_val,max_val), cmap="gray")
  # plt.title(f"multiplied by a={value}")
  # plt.show()
  
  
# plt.show()




# Add salt and pepper noise with probability of 0.04 for salt and pepper
noisy_image = salt_and_pepper_noise(I1, 0.10, 0.10)

# a = (0.2, 0.7)
# kernel_edge_sharpen = delta - a[0] * laplcian

# plt.figure(figsize=(16,8));
# plt.subplot(1, 3, 1)
# plt.title("orginal")
# plt.imshow(I1, cmap="gray")
# plt.subplot(1, 3, 2)
# plt.imshow(convolve_clip(noisy_image,kernel_edge_sharpen,min_val,max_val), cmap="gray")
# plt.title(f"noisy multiplied by a={a[0]}")

# kernel_edge_sharpen = delta - a[1] * laplcian
# plt.subplot(1, 3, 3)
# plt.imshow(convolve_clip(noisy_image,kernel_edge_sharpen,min_val,max_val), cmap="gray")
# plt.title(f"noisy multiplied by a={a[1]}")
# plt.show()




a = (0.2, 0.7)
kernel_edge_sharpen = delta - a[0] * laplcian

# plt.figure(figsize=(16,8));
# plt.subplot(2, 2, 1)
# plt.title("orginal")
# plt.imshow(I1, cmap="gray")

plt.subplot(2, 2, 1)
noisy1 = convolve_clip(noisy_image,kernel_edge_sharpen,min_val,max_val)
plt.imshow(noisy1, cmap="gray")
plt.title(f"noisy multiplied by a={a[0]}")


# 2*2 relatively to 4% S&P noise is 2*2 
# Apply median filtering with a kernel size of 3x3
filtered_image1 = cv2.medianBlur(noisy1.astype(np.uint8), 11)
plt.subplot(2, 2, 2)
plt.imshow(filtered_image1, cmap="gray")
plt.title(f"after median filter 2*2")
plt.show()




kernel_edge_sharpen = delta - a[1] * laplcian
# plt.subplot(2, 2, 3)
# noisy2= convolve_clip(noisy_image,kernel_edge_sharpen,min_val,max_val)
# plt.imshow(noisy2, cmap="gray")
# plt.title(f"noisy multiplied by a={a[1]}")



# # 2*2 relatively to 4% S&P noise 
# # Apply median filtering with a kernel size of 5x5
# filtered_image2 = cv2.medianBlur(noisy2.astype(np.uint8), 5)
# plt.subplot(2, 2, 4)
# plt.imshow(filtered_image2, cmap="gray")
# plt.title(f"after median filter 5*5")





# plt.show()


# a = (0.2, 0.7)
# kernel_edge_sharpen1 = delta - a[0] * laplcian
# kernel_edge_sharpen2 = delta - a[1] * laplcian


# # Generate shot noise
# gamma = 300
# shot_noise = np.random.poisson(gamma, size=I1.shape)
# plt.subplot(1, 3, 1)
# plt.title("orginal")
# plt.imshow(I1, cmap="gray")
# plt.subplot(1, 3, 2)
# plt.imshow(I1+shot_noise, cmap="gray")
# plt.title(f"shot noise with gamma={gamma}")

# noisy = I1+shot_noise

# # Apply median filtering with a kernel size of 3x3
# filtered_image1 = cv2.medianBlur(noisy.astype(np.uint8), 3)
# plt.subplot(1, 3, 3)
# plt.imshow(filtered_image1, cmap="gray")
# plt.title(f"after median filter={gamma}")
# plt.show()

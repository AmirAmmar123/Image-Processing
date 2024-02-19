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

def convolve_clip(Img, kernel, min_val, max_val):
  convolved_image = convolve2d(Img, kernel_edge_sharpen, mode='same')  # Perform 2D convolution
  # Clip the values
  clipped_data = np.clip(convolved_image, min_val, max_val)
  return clipped_data

I1 = cv2.imread('./imgs/Inigo.jpg', cv2.IMREAD_GRAYSCALE)

a = [x*20 for x in range(1,11)] 

# Define the clipping range
min_val = 0
max_val = 255


for index, value in enumerate(a):
  kernel_edge_sharpen = delta - value * laplcian
  plt.subplot(1, 2, 1)
  plt.title("orginal")
  plt.imshow(I1, cmap="gray")
  plt.subplot(1, 2, 2)
  plt.imshow(convolve_clip(I1,kernel_edge_sharpen,min_val,max_val), cmap="gray")
  plt.title(f"multiplied by a={value}")
  plt.show()
  
  
plt.show()









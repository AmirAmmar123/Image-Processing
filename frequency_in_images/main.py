import numpy as np
import cv2
import matplotlib.pyplot as plt

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
  
  

def compare_images(img1,img2, **args):
  # convert each of the RGB chnnels to the correct dynamic range:
  img1 = (((img1-np.min(img1))/np.max(img1))* 255).astype(int);
  img2 = (((img2-np.min(img2))/np.max(img2))* 255).astype(int);
  # Plot:
  plt.figure(figsize=(16,8))
  plt.subplot(121);
  plt.imshow(img1, cmap="gray")
  plt.title(args['title1'])
  plt.xlabel(args['xlabel1'])
  plt.ylabel(args['ylabel1'])
  plt.subplot(122)
  plt.imshow(abs(img2), cmap="gray")
  plt.title(args['title2'])
  plt.xlabel(args['xlabel2'])
  plt.ylabel(args['ylabel2'])
  if( args['save'] ):
    plt.savefig(args['saveSrc'])
  plt.show()


def logarithmic_display_of_image(img):
  return np.log10(np.abs(img)+1)

# Load image in a gray scale format
I1 = cv2.imread('../imgs/Uma.jpg', cv2.IMREAD_GRAYSCALE)

# Perfrom the Furier tranform and shift the result
I1_f = np.fft.fft2(I1);
fshift = np.fft.fftshift(I1_f);
# Convert to log scale and present the output
log_image = logarithmic_display_of_image(fshift);

# compare_images(I1,log_image, title1='Orginal', 
#                 title2='Uma.jpg shifted log magnitude of amplitude',
#                 xlabel1='size', ylabel1='size',
#                 xlabel2='u',
#                 ylabel2='v',
#                 saveSrc='./frequency_in_images/outPut.jpg',
#                 save=True);




height, width = I1.shape
x_partition = int(round(0.05 * width))
y_partition = int(round(0.05 * height))

# Partition the x-axis and y-axis of the Fourier transform
x_axis_sorted = np.sort(I1_f, axis=1)
y_axis_sorted = np.sort(I1_f, axis=0)



# Create an output image initialized with zeros
output_image_fft = np.zeros(I1_f.shape)

# Take the magnitude of complex values and assign them to output_image_fft
output_image_fft[:, :x_partition] = np.abs(x_axis_sorted[:, :x_partition])
output_image_fft[:, -x_partition:] = np.abs(x_axis_sorted[:, -x_partition:])
output_image_fft[:y_partition, :] = np.abs(y_axis_sorted[:y_partition, :])
output_image_fft[-y_partition:, :] = np.abs(y_axis_sorted[-y_partition:, :])

output_image_fft_shift = np.fft.fftshift(output_image_fft)
# plt.imshow(logarithmic_display_of_image(output_image_fft_shift ),cmap="gray")
# plt.show()


output_image_SHIFTED_back = np.fft.fftshift(output_image_fft_shift)
output_image_restores= np.fft.ifft2(output_image_SHIFTED_back)


# # Plot the original image and the transformed image
disply = logarithmic_display_of_image(output_image_fft_shift )
compare_images(disply, output_image_restores, 
                title1='5% lowest frequencies of the image in the x and y axis', 
                title2='Inverse Fourier Transformed of 5% lowest frequencies of the image in the x and y axis',
                xlabel1='Width', ylabel1='Height',
                xlabel2='Frequency (u)', ylabel2='Frequency (v)',
                saveSrc='./outPut2.jpg',
                save=True)




x_axis_sum = np.sum(I1_f, axis=1)
y_axis_sum = np.sum(I1_f, axis=0)




# Assuming you have already computed the sums along the x-axis and y-axis
# x_axis_sum and y_axis_sum contain the sums of values along each row and column, respectively

# Sort the sums along the x-axis and y-axis
sorted_x_indices = np.argsort(x_axis_sum)[::-1]  # Sort in descending order
sorted_y_indices = np.argsort(y_axis_sum)[::-1]  # Sort in descending order

# Calculate the number of columns and rows corresponding to the top 5%
top_5_percent_columns = int(0.05 * len(sorted_x_indices))
top_5_percent_rows = int(0.05 * len(sorted_y_indices))

# Select the top 5% dominant columns and rows
top_5_percent_dominant_columns = sorted_x_indices[:top_5_percent_columns]
top_5_percent_dominant_rows = sorted_y_indices[:top_5_percent_rows]

# Now top_5_percent_dominant_columns contains the indices of the top 5% dominant columns
# and top_5_percent_dominant_rows contains the indices of the top 5% dominant rows



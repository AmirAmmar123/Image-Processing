import numpy as np
import cv2
import matplotlib.pyplot as plt

def present_image(img, title='', xlabel = '', ylabel = ''):
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
I1 = cv2.imread('./imgs/Uma.jpg', cv2.IMREAD_GRAYSCALE)

# Perfrom the Furier tranform and shift the result
I1_f = np.fft.fft2(I1);
fshift = np.fft.fftshift(I1_f);
# Convert to log scale and present the output
# log_image = logarithmic_display_of_image(fshift);

# compare_images(I1,log_image, title1='Orginal', 
#                 title2='Uma.jpg shifted log magnitude of amplitude',
#                 xlabel1='size', ylabel1='size',
#                 xlabel2='u',
#                 ylabel2='v',
#                 saveSrc='./frequency_in_images/outPut.jpg',
#                 save=True);





# # Create an output image initialized with zeros
# binarymask = np.zeros(I1_f.shape)
# R,C = binarymask.shape
# binarymask[ int(0.45 * R) :  int( 0.55 * R), : ] = 1 
# # present_image(binarymask)
# binarymask[ : , int( 0.45 * C) : int( 0.55 * C)] = 1 
# # present_image(binarymask)
# outPut = fshift * binarymask 

# output_image_fft_shift = np.fft.ifftshift(outPut)
# # plt.imshow(logarithmic_display_of_image(output_image_fft_shift ),cmap="gray")
# # plt.show()



# output_image_restores= np.fft.ifft2(output_image_fft_shift)


# # # Plot the original image and the transformed image
# disply = logarithmic_display_of_image(output_image_fft_shift )
# compare_images(logarithmic_display_of_image(outPut), output_image_restores, 
#                 title1='5% lowest frequencies of the image in the x and y axis', 
#                 title2='Inverse Fourier Transformed of 5% lowest frequencies of the image in the x and y axis',
#                 xlabel1='Width', ylabel1='Height',
#                 xlabel2='Frequency (u)', ylabel2='Frequency (v)',
#                 saveSrc='./outPut2.jpg',
#                 save=True)




# Calculate sum of x-axis and y-axis of the Fourier transform
sum_x_axis = np.sum(I1_f, axis=0)
sum_y_axis = np.sum(I1_f, axis=1)

# Sort the array while remembering the index of the original array
sorted_indices_x= sorted(range(len(sum_x_axis)), key=lambda i: sum_x_axis[i])
sorted_indices_y= sorted(range(len(sum_y_axis)), key=lambda i: sum_y_axis[i])

highest_five_percent_sum_amplitude_x_axis_indices =  sorted_indices_x[int(np.ceil(len(sum_x_axis) * 0.95)):]
highest_five_percent_sum_amplitude_y_axis_indices =  sorted_indices_y[int(np.ceil(len(sum_y_axis) * 0.95)):]

mask = np.zeros(I1.shape)
mask[highest_five_percent_sum_amplitude_x_axis_indices,:] = 1
mask[:,highest_five_percent_sum_amplitude_y_axis_indices] = 1



outPut = fshift * np.fft.ifftshift(mask) 


output_image_fft_shift = np.fft.ifftshift(outPut)

output_image_restores= np.fft.ifft2(output_image_fft_shift)





compare_images(logarithmic_display_of_image(mask), output_image_restores, 
                title1='5% highest five percent sum amplitude x and y axis', 
                title2='Image restored',
                xlabel1='Width', ylabel1='Height',
                xlabel2='Frequency (u)', ylabel2='Frequency (v)',
                saveSrc='./outPut2.jpg',
                save=True)


# Assuming I1_f is the frequency representation of the image

# Flatten the image to a 1D array
flattened_image = I1_f.flatten()

# Sort the pixel values
sorted_pixels = np.sort(flattened_image)

top_10_percent_freq_values = sorted_pixels[int(np.ceil(len(sorted_pixels) * 0.9)):]

binarymask = np.zeros(I1_f.shape)


for x in range(0, I1_f.shape[0]):
  for y in range(0, I1_f.shape[1]):
    if I1_f[x,y] >= top_10_percent_freq_values[0] :
      binarymask[x,y] = 1


outPut = fshift * np.fft.ifftshift(binarymask) 

output_image_restores= np.fft.ifft2(np.fft.ifftshift(outPut))



# # # Plot the original image and the transformed image
# compare_images(logarithmic_display_of_image(outPut), output_image_restores, 
#                 title1='10% dominant bidimensional frequencies of the image', 
#                 title2='restored image',
#                 xlabel1='Width', ylabel1='Height',
#                 xlabel2='Frequency (u)', ylabel2='Frequency (v)',
#                 saveSrc='./outPut2.jpg',
#                 save=True)


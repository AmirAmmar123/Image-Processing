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


# Load image in a gray scale format
CAT_I = cv2.imread('./imgs/cat.jpg', cv2.IMREAD_GRAYSCALE)
# Load image in a gray scale format
ANNA_I = cv2.imread('./imgs/Anna.jpg', cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(16,8));
# plt.subplot(1,2,1)
# plt.imshow(CAT_I, cmap="gray");
# plt.title('/imgs/cat.jpg');

# plt.subplot(1,2,2)
# plt.imshow(ANNA_I, cmap="gray");
# # plt.title('./imgs/Anna.jpg');

# plt.show()

def logarithmic_display_of_image(img):
  return np.log10(np.abs(img)+1)

# Perfrom the Furier tranform and shift the result
CAT_Img_f = np.fft.fft2(CAT_I);
CAT_Img_f_shifted = np.fft.fftshift(CAT_Img_f);
phase_CAt = np.angle(CAT_Img_f_shifted)
# phase_CAt = np.angle(CAT_Img_f_shifted)
log_CAT_Img_f_shifted = logarithmic_display_of_image(CAT_Img_f_shifted)
# AMPl_CAT = np.abs(CAT_Img_f_shifted)
AMPl_CAT = np.abs(log_CAT_Img_f_shifted)
# present_image(AMP_CAT, "cat image amplitude", 'u', '|F(u)|');


# Perfrom the Furier tranform and shift the result
ANNA_Img_f = np.fft.fft2(ANNA_I);
ANNA_Img_f_shifted = np.fft.fftshift(ANNA_Img_f);
phase_anna = np.angle(ANNA_Img_f_shifted)
log_ANNA_Img_f_shifted = logarithmic_display_of_image(ANNA_Img_f_shifted)
# AMPl_ANNA = np.abs(ANNA_Img_f_shifted)
AMPl_ANNA = np.abs(log_ANNA_Img_f_shifted)
plt.figure(figsize=(16,8));
plt.subplot(2,2,1)
plt.imshow(CAT_I, cmap="gray");
plt.title('/imgs/cat.jpg');
plt.subplot(2,2,2)
plt.imshow(AMPl_CAT, cmap="gray");
plt.title("cat image amplitude");
plt.xlabel('u')
plt.ylabel('|F(u)|')


plt.subplot(2,2,3)
plt.imshow(ANNA_I, cmap="gray");
plt.title('/imgs/Anna.jpg');
plt.subplot(2,2,4)
plt.imshow(AMPl_ANNA, cmap="gray");
plt.title("Anna image amplitude");
plt.xlabel('u')
plt.ylabel('|F(u)|')



# plt.show()


# # Multiply the amplitude spectrum of Anna's image by the phase spectrum of the cat image
# new_out_put =np.abs(ANNA_Img_f_shifted) * np.exp(1j * phase_CAt)

# # Perform the inverse Fourier transform to obtain the new image
# Shifted_back = np.fft.ifftshift(new_out_put)
# new_out_put_restored = np.fft.ifft2(Shifted_back)

# # Display the resulting image
# present_image(np.abs(new_out_put_restored), "New Image")

# Generate random values within the specified range
random_values_for_ampl = np.random.uniform(np.min(AMPl_ANNA), np.max(AMPl_ANNA), size=(512, 512))
random_values_for_phase = np.random.uniform(np.min(phase_anna), np.max(phase_anna), size=(512, 512))


ampl_rand_phase_anna = random_values_for_ampl* np.exp(1j * phase_anna)
phase_rand_ampl_anna = np.abs(ANNA_Img_f_shifted) * np.exp(1j * random_values_for_phase)

plt.figure(figsize=(16,8));
plt.subplot(1,2,1)
plt.imshow(np.abs(ampl_rand_phase_anna), cmap="gray");
plt.title('randome amplitude with anna phase');
plt.subplot(2,2,2)
plt.imshow(np.abs(phase_rand_ampl_anna), cmap="gray");
plt.title("randome phase with anna amplitude");
plt.show()

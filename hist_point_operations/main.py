import skimage
from skimage import io,exposure
import numpy as np
import matplotlib.pyplot as plt

def ploth(hist,b_c, title):
    plt.figure(figsize=(10, 4))
    plt.bar(b_c, hist, width=1)
    plt.title(title)
    plt.xlabel('gray-levels')
    plt.ylabel('number of pixels')

def PowerLawTransformation(r, Y = 1 ,C = 1, strech_in_range = False, T = 0):
    # Power law transformation function 
    # r image before PowerLawTransformation 
    # Y,C constants 
    # return s image after PowerLawTransformation 
    indices1 = np.where( r >= T)
    indices2 = np.where( r < T )
    none_zero_one_img = False
    if r.max() > 1 :
        none_zero_one_img = True
        r = r/255 # if the dynamic domain i between 0 and and 255, normalize the pixels to the range 0 to 1
      
    if not strech_in_range :
      s = C * np.power(r, Y)
    else:
      s = np.zeros(r.shape)
    
      s[indices2]= r[indices2]
      s[indices1]= C * np.power(r[indices1], Y)
       
    if none_zero_one_img :
        s = np.round(s * 256)
    return s

def histEqualization(origin_img):

    
    none_zero_one_img = False
    float_image = origin_img
    if origin_img.max() > 1 :
        none_zero_one_img = True
        float_image = origin_img/255.0 # if the dynamic domain i between 0 and and 255, normalize the pixels to the range 0 to 1 
        
    # org_hist , gray_scales = exposure.histogram(origin_img)
    # ploth(org_hist, gray_scales, "orginal")
    
    
    
    float_hist, gray_scales = exposure.histogram(float_image) 
    total_pixels=float_hist.sum()
    norm_hist = float_hist / total_pixels


    cdf = np.cumsum(norm_hist)
    cdf_hest , bins = exposure.histogram(cdf*255)
    plt.figure(figsize=(10, 4))
    plt.bar(bins, cdf_hest, width=1)
    plt.title("commulative function")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    
    HE_img = cdf[origin_img]
    if none_zero_one_img :
      HE_img=HE_img*255
    equalized_hist, gray_scales = exposure.histogram(HE_img) 
    # ploth(equalized_hist, gray_scales,"equalized")
    return HE_img
    

################## Q1 part A ################# 
r = np.array([0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25 ])
C = 1
Y = 1
# histogram, bins = exposure.histogram(r)
# plt.subplot(1, 2, 1)
# plt.title(f"PowerLawTransformation(r,C={C},Y={Y})")
# plt.bar(bins, histogram, width=1)
# plt.ylabel('number of pixels')
# plt.xlabel('gray-levels')

# Y = 0.7
# C = 2
# result = PowerLawTransformation(r,Y,C)
# histogram, bins = exposure.histogram(result)
# plt.subplot(1, 2, 2)
# plt.title(f"PowerLawTransformation(r,C={C},Y={Y})")
# plt.bar(bins, histogram, width=1)
# plt.ylabel('number of pixels')
# plt.xlabel('gray-levels')
# plt.show()

# for y=1 and c =1 nothing has changed 
# for y=0.5 and c=1 the image has transformed to higher gray scales 
# for y=0.5 and c=2 has scaled each gray level to each pixel twice the amount in the previous result 
# if y < 1 then the image is transformed to higher gray scales 
# if y < a and c > 1 then the image is transformed c * (the histogram of r^gamma)
# for y = 2 and c=1 the gray level has been transformed to the domain [0, 2]
# C is actually scalling the gray level domain from [new_img.min, new_img.max] to c * [new_img.min, new_img.max]
# for C = 1 and y=3 the image transformed to almost zero 
# if y > 1 the image is transformed to lower gray scaled 
# C is scalling the new dynamic domain 
# the function acts like C *( gamma correction )



################## Q1 part B #################
orgImg = io.imread('./imgs/mri_spine.jpg')
# plt.figure(figsize=(6, 6))
# plt.imshow(orgImg,cmap="gray") 
# plt.title("mri_spine.jpg")
# plt.show()

# plt.figure(figsize=(10, 4))
# origin_hist, bin_centers = exposure.histogram(orgImg) # most of the pixels is living close to [0, 50],
# plt.bar(origin_hist, bin_centers)
# plt.title("histogram of mri_spine.jpg")
# plt.xlabel('gray-levels')
# plt.ylabel('number of pixels')
# plt.show()


# arr = [1 for x in range(1,11,1)]
# Ys, Cs = np.arange(0,1,1/10), np.array(arr)
# LUT = zip(Ys,Cs)
# result = np.array([])


# index = 1 
# # plt.subplot(4, 3, index)
# plt.figure(figsize=(16, 8))
# indx = index + 1
# orgHist, bin_centers = exposure.histogram(orgImg)
# plt.title(f"org hist")
# plt.bar(bin_centers, orgHist, width=1)
# plt.ylabel('number of pixels')
# plt.xlabel('gray-levels')
# plt.show()


# for Y,C in LUT:
  
#   if Y != 0 and C != 0 :
#     plt.figure(figsize=(16, 8))
#     result = PowerLawTransformation(orgImg,Y,C)
#     after, bin_centers = exposure.histogram(result)
#     # trying to avoid pixel value 0^0 
#     # plt.subplot(4, 3, index)
#     plt.title(f'Y={Y} C={C}')
#     plt.bar(bin_centers, after, width=1)
#     plt.ylabel('number of pixels')
#     plt.xlabel('gray-levels')
#     plt.show()
#     index = index + 1
#     plt.imshow(result,cmap="gray") 
#     plt.show()
  
  



# best result when y in [0,1] is , y = 0.7 and C = 1 
Y = 0.7 
C = 1 
result = PowerLawTransformation(orgImg,Y,C)

# ploth(after, bin_centers, f'after PowerLawTransformation Y={Y} C={C} mri_spine.jpg')
# plt.figure(figsize=(16,8));
# ax = plt.subplot(121);
# plt.imshow(orgImg,cmap="gray") 
# ax.set_title("before")
# ax = plt.subplot(122)
# result = PowerLawTransformation(orgImg,0.7,1)
# plt.imshow(result,cmap="gray") 
# ax.set_title("after")
# plt.show()

# histOrg, bins_Org = exposure.histogram(orgImg)
# plt.subplot(1, 2,1)
# plt.title("Before PowerLawTransformation")
# plt.bar(bins_Org, histOrg, width=1)
# plt.ylabel('number of pixels')

# histAfter, bins_after = exposure.histogram(result) 
# plt.subplot(1, 2,2)
# plt.title("After PowerLawTransformation")
# plt.bar(bins_after, histAfter, width=1)
# plt.ylabel('number of pixels')

# plt.show()




################## Q1 part C #################



# HE_img = histEqualization(orgImg)
# plt.figure(figsize=(16,8));
# plt.subplot(1, 2,1)
# plt.title("Orginal Image")
# plt.imshow(orgImg,cmap="gray") 
# plt.subplot(1, 2,2)
# plt.imshow(HE_img,cmap="gray") 
# plt.title("After Histogram Equalization")
# plt.show()

HE_img = histEqualization(orgImg)

HistOrg, grayScalesOrg = exposure.histogram(orgImg)
plt.figure(figsize=(16,8));
plt.subplot(1, 2,1)
plt.bar(grayScalesOrg, HistOrg, width=1)
plt.title("Orginal Histogram")
plt.xlabel('gray-levels')
plt.ylabel('number of pixels')

HistHE, grayScalesHE = exposure.histogram(HE_img)
plt.subplot(1, 2,2)
plt.bar(grayScalesHE, HistHE, width=1)
plt.title("Histogram Equalization")
plt.xlabel('gray-levels')
plt.ylabel('number of pixels')
plt.show()

# we did not get histogram equalization 
# in the original picture before histogram equalization, we have approx' X=100000 pixels of gray level of 0
# relatively to the total number of pixels this is the highest ratio 
# (X/182024) * 255  aprrox => [135, 140]
# which mean when applying the commulative function most of the pixels will be transport to  [135, 140]
# so this will be the minimum gray level which most of the pixels will be transport to 
# other columns ratios do not contribute alot to the total ratio, to due to lake of pixels in the spicific gray level higher than zero 
# so we can see that the pixels between  0 < gray-level < 100 has been transformed to  [135, 140] < level < 220
# and yet those pixel will still have have the same ratio, but will be more brighter 
# this happens due to the nature of the cumsum function (comulative function) which first accumalte the total ratios till specific gray level
# so the most gray level ratio contributed to the cumsum function is the  gray level of 0 
# and since all other levels are higher than zero and, they dont have have alot of pixel relatively to 0 gray level 
# will not contribute alot to the the cumsum functio ==> so also they will be effected directly by the  0 gray level  transformation
 


################## Q1 part D #################


T = 50
C= 1
Y =  0.6666
# outImg_partD = PowerLawTransformation(orgImg,Y,C,True,T)
# org_hist , gray_scales = exposure.histogram(orgImg)
# ploth(org_hist, gray_scales, "orginal")

# outImg_hist , gray_scales = exposure.histogram(outImg_partD)
# ploth(outImg_hist, gray_scales, "new image")
# plt.show()


org_hist, gray_scales = exposure.histogram(orgImg)
plt.subplot(1, 2, 1)
plt.title("Original")
plt.bar(gray_scales, org_hist, width=1)
plt.ylabel('Number of pixels')

outImg_partD = PowerLawTransformation(orgImg, Y, C, True, T)
outImg_hist, gray_scales = exposure.histogram(outImg_partD)

plt.subplot(1, 2, 2)
plt.title("Transformed")
plt.bar(gray_scales, outImg_hist, width=1)
plt.ylabel('Number of pixels')

plt.show()

################## Q1 part E #################
Raw = int(np.round(orgImg.shape[0]/3))
Col = int(np.round(orgImg.shape[1]/3))

UP_L = orgImg[:Raw,:Col]
Up_M = orgImg[:Raw, Col:2*Col]
Up_R = orgImg[:Raw, 2*Col:]

Midle_L = orgImg[Raw:2*Raw, :Col]
Midle_M = orgImg[Raw:2*Raw, Col:2*Col]
Midle_R = orgImg[Raw:2*Raw,  2*Col:]

Bottom_L = orgImg[2*Raw:, :Col]
Bottom_M = orgImg[2*Raw:, Col:2*Col]
Bottom_R = orgImg[2*Raw:,  2*Col:]

parts = [UP_L, Up_M, Up_R, Midle_L , Midle_M, Midle_R, Bottom_L, Bottom_M, Bottom_R]
mapper = {0:"UP_L"}
plt.figure(figsize=(16, 8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
mapper = {
    0: "UP_L",
    1: "Up_M",
    2: "Up_R",
    3: "Midle_L",
    4: "Midle_M",
    5: "Midle_R",
    6: "Bottom_L",
    7: "Bottom_M",
    8: "Bottom_R"
}

for index, part in enumerate(parts):
    plt.subplot(3, 3, index+1)
    plt.imshow(part, cmap="gray")
plt.show()
  
for index, part in enumerate(parts):
  hist, bins = exposure.histogram(part)
  plt.subplot(3, 3, index+1)
  plt.title(f"part {mapper[index]}")
  plt.bar(bins, hist, width=1)
  plt.ylabel('number of pixels')
plt.show()

T_gamma_UP_L = (5, 0.7)
T_gamma_UP_M = (2, .7)
T_gamma_Up_R = (100, 1.1)
T_gamma_Midle_L = (0, 1)
T_gamma_Midle_M = (2, .7)
T_gamma_Midle_R =  (2, .7)
T_gamma_Bottom_L = (0, 2,1)
T_gamma_Bottom_M = (2,0.7)
T_gamma_Bottom_R = (2, .7)
Ts = [T_gamma_UP_L,T_gamma_UP_M,T_gamma_Up_R, 
      T_gamma_Midle_L,T_gamma_Midle_M,T_gamma_Midle_R, 
      T_gamma_Bottom_L, T_gamma_Bottom_M,T_gamma_Bottom_R ]
fixedarr = []
for  index, fixMe in  enumerate( zip(parts,Ts) ):
  # plt.subplot(3, 3, index+1)
  parts[index]= PowerLawTransformation(fixMe[0],fixMe[1][1],C,True, fixMe[1][0])
  # plt.imshow(fixMe[0], cmap="gray")
  

# plt.show()

# Concatenate parts horizontally
top_row = np.hstack([parts[0], parts[1], parts[2]])
middle_row = np.hstack([parts[3], parts[4], parts[5]])
bottom_row = np.hstack([parts[6], parts[7], parts[8]])

# Concatenate rows vertically
orgImg_concatenated = np.vstack([top_row, middle_row, bottom_row])

# Show the concatenated image
plt.figure(figsize=(16, 8))
plt.subplot(1,3,1)
plt.imshow(orgImg, cmap="gray")
plt.title("Orginal Image")
plt.subplot(1,3,2)
plt.imshow(outImg_partD, cmap="gray")
plt.title("part D Image")
plt.subplot(1,3,3)
plt.imshow(orgImg_concatenated, cmap="gray")
plt.title("Concatenated Image")


plt.axis('off')
plt.show()


#plot
# plt.figure(figsize=(16,8));
# ax = plt.subplot(121);
# plt.imshow(orgImg,cmap='gray');
# ax.set_title("Original");

# ax = plt.subplot(122)
# plt.imshow(fixed,cmap='gray');
# ax.set_title("Thresholed");
# plt.show()
# Plotting UP_Left


# plt.figure(figsize=(16, 8))

# # Plotting UP_Left
# plt.subplot(3, 3, 1)
# plt.imshow(UP_L, cmap="gray")
# plt.title("UP_Left")

# # Plotting Up_Midle
# plt.subplot(3, 3, 2)
# plt.imshow(Up_M, cmap="gray")
# plt.title("Up_Midle")

# # Plotting Up_Right
# plt.subplot(3, 3, 3)
# plt.imshow(Up_R, cmap="gray")
# plt.title("Up_Right")


# # Plotting Midle_Left
# plt.subplot(3, 3, 4)
# plt.imshow(Midle_L, cmap="gray")
# plt.title(" Midle_Left")

# # Plotting Midle_Midle
# plt.subplot(3, 3, 5)
# plt.imshow(Midle_M, cmap="gray")
# plt.title(" Midle_Left")

# # Plotting Midle_Right
# plt.subplot(3, 3, 6)
# plt.imshow(Midle_R, cmap="gray")
# plt.title(" Midle_Left")


# # Plotting Midle_Left
# plt.subplot(3, 3, 4)
# plt.imshow(Midle_L, cmap="gray")
# plt.title(" Midle_Left")

# # Plotting Midle_Midle
# plt.subplot(3, 3, 5)
# plt.imshow(Midle_M, cmap="gray")
# plt.title(" Midle_Left")

# # Plotting Midle_Right
# plt.subplot(3, 3, 6)
# plt.imshow(Midle_R, cmap="gray")
# plt.title(" Midle_Left")


# # Plotting Bottom_Left
# plt.subplot(3, 3, 7)
# plt.imshow(Bottom_L, cmap="gray")
# plt.title("Bottom_Left")

# # Plotting Bottom_Middle
# plt.subplot(3, 3, 8)
# plt.imshow(Bottom_M, cmap="gray")
# plt.title("Bottom_Middle")

# # Plotting Bottom_Right
# plt.subplot(3, 3, 9)
# plt.imshow(Bottom_R, cmap="gray")
# plt.title("Bottom_Right")


# plt.show()

##################
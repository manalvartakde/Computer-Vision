import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2.aruco as aruco

##### Read images #####
img_path = r""     # Insert image path here
display_img_path  = r""         # Insert display image path here
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display_image = cv2.imread(display_img_path)
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)


##### detecting and reading Aruco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

##### Only corner points are useful (getting them in readable format)
corner = np.reshape(corners,(-1))
corner = corner.astype(int)

##### Defining original image points and perspective transfromation points
(x, y) = (850, 440) # Adjust display_image scaling here )
(tr, tl, br, bl) = ([0+x, 0+y], [display_image.shape[1]-x, 0+y], [display_image.shape[1]-x, display_image.shape[0]-y], [0+x, display_image.shape[0]-y])
display_img_corner_pts = np.float32([tr, tl, br, bl]) # Four corner coordinates of the display image
perspective_shifted_pts = np.float32([[corner[0], corner[1]], [corner[2], corner[3]], [corner[4], corner[5]], [corner[6], corner[7]]])

#################### Perspective transform of display image #######################

# This is a transfirmation matrix to transform our image into different perspective
matrix = cv2.getPerspectiveTransform(display_img_corner_pts, perspective_shifted_pts)
print(matrix)

# perspective of the display image is changed according to the wall viewing angle
transformed_image = cv2.warpPerspective(display_image, matrix, (image.shape[1], image.shape[0])) 
print(transformed_image.shape)
#################### x Perspective transform of display image x #####################

#################################### Masking ############################################
''' 
Bitwise_and -> White(mask) values retained from original image; rest all is filled (0,0,0)
Bitwise_or -> everything *except* White(mask) id retained; rest all(mask) is filled (255,255,255) 
'''
###################### New mask #################

mask1 = np.zeros_like(image, dtype= np.uint8)
mask1[:,:] = (255,255,255)
b, g, r = cv2.split(transformed_image)

########################### Jugad ################################
mask_r = np.zeros_like(r)
mask_r[r != 0] = 255
mask_g = np.zeros_like(g)
mask_g[g != 0] = 255
mask_b = np.zeros_like(b)
mask_b[b != 0] = 255 
##################################################################

mask = cv2.merge((mask_b, mask_g, mask_r))
mask = np.ones_like(mask) * 255 - mask
vertices1 = np.int32([[corner[0], corner[1]], [corner[2], corner[3]], [corner[4], corner[5]], [corner[6], corner[7]]])

# cv2.fillPoly(mask1, [vertices1], (0,0,0))
# mask1 = cv2.warpPerspective(mask1, matrix, (image.shape[1], image.shape[0]))
# Bitwise_and retains pixel values in original image where the mask values are nonzero and fills rest with (0,0,0)
masked_image1 = cv2.bitwise_and(image, mask)  # insted of barcode there, its filled with zeros 

##################### x New mask x ####################
#################################### x Masking x ##########################################

# Pixel values are added fromt he two images
add_image =  cv2.add(masked_image1, transformed_image)
plt.imshow(add_image)
plt.show()
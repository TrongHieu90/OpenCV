import cv2
import numpy as np

apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')

print(apple.shape)
print(orange.shape)

apple_orange = np.hstack((apple[:, :256], orange[:, 256:]))

# generate Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]

for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)
    
    
# generate Gaussian pyramid for orange
orange_copy = orange.copy()
gp_orange = [orange_copy]

for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# generate Laplacian pyramid for apple
apple_copy = gp_apple[5]
lp_apple = [apple_copy]

for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    lapacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
    lp_apple.append(lapacian)
    
# generate Laplacian pyramid for orange
orange_copy = gp_orange[5]
lp_orange = [orange_copy]

for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    lapacian = cv2.subtract(gp_orange[i-1], gaussian_expanded)
    lp_orange.append(lapacian)
    

for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_apple.append(orange_copy)
    
# now add left and right haves of images in each level

apple_orange_pyramid = []

n = 0
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    n +=1
    cols, rows, channel = apple_lap.shape
    lapacian = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):]))
    apple_orange_pyramid.append(lapacian)
    
#reconstruct image
apple_orange_reconstruct = apple_orange_pyramid[0]

for i in range(1,6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)

cv2.imshow('apple', apple)
cv2.imshow('orange', orange)
cv2.imshow('apple_orange', apple_orange)
cv2.imshow('apple_orange_mix', apple_orange_reconstruct)
#cv2.imshow('image', image)
k  = cv2.waitKey(0) & 0xFF #mash for 64 bit machine

if k == 27: #escape key
    cv2.destroyAllWindows()



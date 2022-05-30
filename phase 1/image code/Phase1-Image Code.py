import cv2
import numpy as np
import matplotlib.pylab as plt

def drow_the_lines(image, lines):
    img = np.copy(image)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=30)
    image = cv2.addWeighted(image, 1.25, blank_image, 3, 2)
    return image

def region_of_interest(image, verticess):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask, verticess, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread("pic1.jpg.jpeg")
print(image.shape)
h = image.shape[0]
w = image.shape[1]
region_of_interest_ver = [(0, h), (w/2, h/2), (w, h)]
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 200 , 500)
cropped_image = region_of_interest(canny, np.array([region_of_interest_ver], np.int32), )
lines = cv2.HoughLinesP(cropped_image,rho=10,theta=np.pi/180,threshold=350,lines=np.array([]),
                        minLineLength=150,maxLineGap=150)

image_with_lines = drow_the_lines(image, lines)
plt.imshow(image_with_lines)
plt.show()

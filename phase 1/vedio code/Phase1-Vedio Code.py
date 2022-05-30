import cv2
import numpy as np

def drow_the_lines(Image, lines):
    Image = np.copy(Image)
    blank_image = np.zeros((Image.shape[0], Image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for X, Y, X1, Y1 in line:
            cv2.line(blank_image, (X, Y), (X1, Y1), (0, 250, 0), thickness=10)
    Image = cv2.addWeighted(Image, 1.25, blank_image, 3, 2)
    return Image

def region_of_interest(Image, vertices):
    mask = np.zeros_like(Image)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(Image, mask)
    return masked_image

def process(Image):
    print(Image.shape)
    h = Image.shape[0]
    w = Image.shape[1]
    region_of_interest_ver = [(0, h), (w/2, h/2), (w, h)]
    grey = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(grey, 200, 500)
    cropped = region_of_interest(canny, np.array([region_of_interest_ver], np.int32), )
    lines = cv2.HoughLinesP(cropped, rho=6,theta=np.pi/180, threshold=200, lines=np.array([]),
                            minLineLength=50,maxLineGap=75)

    image_with_lines = drow_the_lines(Image, lines)
    return image_with_lines

cap = cv2.VideoCapture("Video.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow(" frame ", frame)
    if cv2.waitKey(75) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
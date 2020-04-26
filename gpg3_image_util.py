import cv2

def process_image(image):
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = bw_image.shape

    # assuming the camera is aimed at a 45 degree angle and mounted
    # on the top platform of the GPG, extract only the region right in
    # front of the camera
    startY = int(H * 0.75)
    endY = H
    startX = int(W * 0.2)
    endX = int(W * 0.8)
    roi = bw_image[startY:endY, startX:endX]

    # Threshold the image
    (T, thresh) = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

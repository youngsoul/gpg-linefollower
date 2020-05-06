import cv2

def process_image(image):
    """
    Take in a RGB image from the RPI camera.
    Crop the image to only the part of the image that is in front of the camera/car
    Change the image to gray scale, and then threshold to get a black and white image.
    :param image:
    :type image:
    :return: thresh black and white image, roi which is the cropped RGB image
    :rtype:
    """
    (H, W, C) = image.shape

    # assuming the camera is aimed at a 45 degree angle and mounted
    # on the top platform of the GPG, extract only the region right in
    # front of the camera
    startY = int(H * 0.75)
    endY = H
    startX = int(W * 0.2)
    endX = int(W * 0.8)
    roi = image[startY:endY, startX:endX]
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    (T, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh, roi

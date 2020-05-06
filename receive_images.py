"""
When the car is driving by model, the car can send images to this server for display.

This script will display the image from the and indicate if the prediction was
-Left
-Right
-Straight

We can then see what the car sees, and that the model tells the car to do.
"""
import cv2
from imagezmq.imagezmq import ImageHub
import socket
import signal
import sys

# ImageHub is the class that allows for receiving images from ZMQ
image_hub = ImageHub()


def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    if image_hub:
        image_hub.clean_up()

    sys.exit()


signal.signal(signal.SIGINT, signal_handler)

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print(f"Computer Name: {hostname}")
print(f"IP: {IPAddr}")

# draw a circle on the image to indicate the predicted value
# from the GoPiGo
left_coordinates = (20, 40)

straight_coordinates = (96, 40)

right_coordinates = (172, 40)

# Radius of circle
radius = 20

# Director circle color in BGR
left_color = (255, 0, 0)
straight_color = (0, 255, 0)
right_color = (0, 0, 255)

while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    print(rpi_name)
    if rpi_name == 'left':
        image = cv2.circle(image, left_coordinates, radius, left_color, thickness=-1)
    elif rpi_name == 'straight':
        image = cv2.circle(image, straight_coordinates, radius, straight_color, thickness=-1)
    elif rpi_name == 'right':
        image = cv2.circle(image, right_coordinates, radius, right_color, thickness=-1)

    # The protocol is a request/response so we must respond with a message.
    image_hub.send_reply(b'OK')

    cv2.imshow("GoPiGo", image)
    cv2.waitKey(1)

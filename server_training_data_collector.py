"""
"""
import cv2
from imagezmq.imagezmq import ImageHub
import socket
import signal

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print(f"Computer Name: {hostname}")
print(f"IP: {IPAddr}")

# stop, forward, backwards, exit, straight, left, right
car_command = b'stop'

image_hub = ImageHub()

def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    image_hub.clean_up()

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':

    while True:  # show streamed images until Ctrl-C
        rpi_name, image = image_hub.recv_image()
        cv2.imshow(rpi_name, image)  # 1 window for each RPi
        key = cv2.waitKey(1)
        if key == ord('s'):
            print("save image...")
        elif key == ord('1'):
            car_command = b'left'
        elif key == ord('2'):
            car_command = b'straight'
        elif key == ord('3'):
            car_command = b'right'
        elif key == ord('0'):
            car_command = b'stop'
        elif key == ord('9'):
            car_command = b'forward'
        elif key == ord('8'):
            car_command = b'backward'
        elif key == ord('x'):
            car_command = b'exit'

        if car_command is not None:
            image_hub.send_reply(car_command)
        else:
            image_hub.send_reply(b'OK')

        car_command = None

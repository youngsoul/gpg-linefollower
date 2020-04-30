from joblib import load
import easygopigo3
import signal
import time
import sys
import argparse
from imagezmq.asyncimagesender import AsyncImageSender
from imutils.video import VideoStream
import imutils as imutils
import socket
from gpg3_image_util import process_image
from pathlib import Path
import cv2

import logging
logger = logging.getLogger("AsyncImageSender")
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


gpg = easygopigo3.EasyGoPiGo3()
# gpg.set_speed(100)
# sleep_between_image_sends = 0.1

WHEEL_SPEED_CONSTANT = 40
# (left multiplier, right multiplier)
left_turn = (0.5, 1.5)
right_turn = (1.5, 0.5)


directions = ["left", "straight", "right"]

def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

saved_image_index = 0
def save_image(image_to_save):
    global saved_image_index
    print(f"Write File: ./route_images/image_{saved_image_index}.jpg")
    cv2.imwrite(f"./route_images/image_{saved_image_index}.jpg", image_to_save)
    saved_image_index += 1


def setup_async_image_sender():
    # get the host name, initialize the video stream, and allow the
    # camera sensor to warmup
    rpiName = socket.gethostname()

    video_stream = VideoStream(usePiCamera=True).start()

    async_image_sender = AsyncImageSender(server_name=rpiName, server_ip=server_ip, port=5555, send_timeout=10,
                                           recv_timeout=10, show_frame_rate=10, backlog=3)

    return video_stream, async_image_sender

def load_model():
    model = load('./rpi_gpg3_line_follower_model.sav')
    return model

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server-ip", required=False, default='192.168.1.208',
                    help="ip address of the server to which the client will connect")
    ap.add_argument("-r", "--rotate", required=False, default=0, help="Rotate the image by the provided degrees")
    ap.add_argument("--turn-degrees", required=False, default=5, type=int, help="Degress to turn")
    ap.add_argument("--blocking", required=False, default=0, type=int, help="1-blocking=True in GPG calls, 0-blocking is false")
    ap.add_argument("--save-every-n", required=False, default=0, type=int, help="Save every n images while driving route. 0=none saved")

    args = vars(ap.parse_args())
    server_ip = args['server_ip']
    rotation = float(args['rotate'])
    turn_degrees = args['turn_degrees']
    blocking = False if args['blocking'] == 0 else True
    every_n_route_images = args['save_every_n']
    if every_n_route_images > 0:
        Path('./route_images').mkdir(parents=True, exist_ok=True)

    video_stream, async_image_sender = setup_async_image_sender()

    model = load_model()
    loop_count = 0
    while True:

        s = time.time()
        frame = video_stream.read()
        if frame is not None:
            if rotation != 0:
                frame = imutils.rotate(frame, rotation)

            image = process_image(frame)
            if every_n_route_images > 0 and loop_count % every_n_route_images == 0:
                save_image(image)

            flatten_image = image.flatten()
            # async_image_sender.send_frame_async(image)

            sm = time.time()
            prediction = model.predict([flatten_image])
            em = time.time()
            print(f"Predict time: {(em-sm)} seconds")

            direction = prediction[0]
            print(f"Predicted direction: {directions[direction]}")

            if direction == 0: # left
                # gpg.turn_degrees(-turn_degrees, blocking=blocking)
                # gpg.forward()
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
            elif direction == 1: #straight
                # gpg.forward()
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
            elif direction == 2: # right
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])

                # gpg.turn_degrees(turn_degrees, blocking=blocking)
                # gpg.forward()
            else:
                print(f"Unknown direction: {direction}")


        e = time.time()
        print(f"Loop Time: {(e-s)} seconds")
        # time.sleep(sleep_between_image_sends)

    gpg.reset_all()



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

# Create instance of the GoPiGo class
gpg = easygopigo3.EasyGoPiGo3()

# Initialize the button to start/stop the car
# This is using 'Analog/Digital2' on the GoPiGo board.
go_button = gpg.init_button_sensor("AD2")
RELEASED = 0
PRESSED = 1
button_state = RELEASED

# Initialize the state of the car in the stopped state
CAR_GO = 1
CAR_STOP = 0
go_state = CAR_STOP

# Set the speed.  This is a value that can be experimented with.  You have to
# maintain the right balance between speed, turning rate and inference speed.
WHEEL_SPEED_CONSTANT = 40

# turn rate.  These values determine how aggresive the turn will be.
# it does this by changing the power to the wheels
# left_turn = (leftwheelspeedfactor, rightwheelspeedfactor)
# for a left turn we reduce the speed of the left wheel and increase the speed of the right wheel.  This will
# cause the car to turn or rotate to the left.  These are values that have to be experimented with.
# (left multiplier, right multiplier)
left_turn = (0.4, 1.6)
right_turn = (1.6, 0.4)

# Friendly names for the direction predictions
directions = ["left", "straight", "right"]

def check_car_state():
    """
    Check to see if the car should go or stop
    :return:
    :rtype:
    """
    global button_state, go_state
    p = go_button.read()
    if button_state == RELEASED and p == 1:
        button_state = PRESSED
        if go_state == CAR_STOP:
            go_state = CAR_GO
        else:
            go_state = CAR_STOP

    if p == 0:
        button_state = RELEASED


def signal_handler(sig, frame):
    """
    Exit gracefully
    :param sig:
    :type sig:
    :param frame:
    :type frame:
    :return:
    :rtype:
    """
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

saved_image_index = 0
def save_image(image_to_save):
    """
    Sometimes it is helpful to save the images the car is seeing while it is driving to help collect
    training images or to debug the model.
    :param image_to_save:
    :type image_to_save:
    :return:
    :rtype:
    """
    global saved_image_index
    print(f"Write File: ./route_images/image_{saved_image_index}.jpg")
    cv2.imwrite(f"./route_images/image_{saved_image_index}.jpg", image_to_save)
    saved_image_index += 1


def setup_async_image_sender():
    """
    Initialize VideoStream and Async Image Sender
    :return:
    :rtype:
    """
    # get the host name, initialize the video stream, and allow the
    # camera sensor to warmup
    rpiName = socket.gethostname()

    video_stream = VideoStream(usePiCamera=True).start()

    async_image_sender = AsyncImageSender(server_name=rpiName, server_ip=server_ip, port=5555, send_timeout=10,
                                           recv_timeout=10, show_frame_rate=10, backlog=3)

    time.sleep(0.2) # give everthing a chance to settle out
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
    ap.add_argument("--save-every-n", required=False, default=0, type=int, help="Save every n images while driving route. 0=none saved")
    ap.add_argument("--send-images", required=False, default=0, type=int, help="1-send image to server, 0[default]-do not send image to server")

    args = vars(ap.parse_args())
    send_images = args['send_images']
    server_ip = args['server_ip']
    rotation = float(args['rotate'])
    every_n_route_images = args['save_every_n']
    if every_n_route_images > 0:
        Path('./route_images').mkdir(parents=True, exist_ok=True)

    video_stream, async_image_sender = setup_async_image_sender()

    async_image_sender.run_in_background()
    time.sleep(2)

    model = load_model()
    loop_count = 0
    while True:
        check_car_state()
        if go_state == CAR_STOP:
            gpg.stop()
            time.sleep(0.5)
            continue

        s = time.time()
        frame = video_stream.read()
        if frame is not None:
            if rotation != 0:
                frame = imutils.rotate(frame, rotation)

            # get a cropped, black and white image and the rgb roi
            image, roi = process_image(frame)

            # check to see if we need to save the raw image
            if every_n_route_images > 0 and loop_count % every_n_route_images == 0:
                save_image(image)

            # flatten the image so we can run it through the ML model
            flatten_image = image.flatten()

            # from the flattened image, predict which direction to go
            sm = time.time()
            prediction = model.predict([flatten_image])
            em = time.time()
            # print(f"Predict time: {(em-sm)} seconds")

            direction = prediction[0]
            # print(f"Predicted direction: {directions[direction]}")

            # the loop is pretty fast so only send one image for every few thousand.
            # I found 4000 to be a good number and the images on the server
            # looked good.
            if send_images == 1 and loop_count % 4000 == 0:
                async_image_sender.server_name = directions[direction]
                async_image_sender.send_frame_async(roi)

            # based on the prediction, change the wheel power to make turns
            if direction == 0: # left
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
            elif direction == 1: #straight
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
            elif direction == 2: # right
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
            else:
                print(f"Unknown direction: {direction}")


        e = time.time()
        # print(f"Loop Time: {(e-s)} seconds")

    gpg.reset_all()



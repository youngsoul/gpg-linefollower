import sys
sys.path.insert(0, '../')  # imagezmq.py is in ../imagezmq
import time
from imagezmq.asyncimagesender import AsyncImageSender
from imutils.video import VideoStream
import imutils as imutils
import socket
import argparse
from gpg3_image_util import process_image


"""
Test file that will create 2 AsyncImageSender classes and send an image to the specified server and port

"""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=False, default='192.168.1.208',
                help="ip address of the server to which the client will connect")
ap.add_argument("-r", "--rotate", required=False, default=0, help="Rotate the image by the provided degrees")

args = vars(ap.parse_args())
server_ip = args['server_ip']
rotation = float(args['rotate'])

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()

video_stream = VideoStream(usePiCamera=True).start()

# async_image_sender1 = AsyncImageSender(server_name=rpiName, server_ip=server_ip, port=5555, send_timeout=10, recv_timeout=10, show_frame_rate=10)

async_image_sender1 = AsyncImageSender(server_name=rpiName, server_ip=server_ip, port=5555, send_timeout=10,
                                      recv_timeout=10, show_frame_rate=10, backlog=3)
async_image_sender1.run_in_background()
time.sleep(2)

image_count = 0

print("Press ctrl-c to stop image sending")
sleep_time = 0.25

while True:
    frame = video_stream.read()
    if frame is not None:
        if rotation != 0:
            frame = imutils.rotate(frame, rotation)

        # response = async_image_sender1.send_frame_immediate(frame)
        image2, roi = process_image(frame)

        response = async_image_sender1.send_frame_async(roi)
        # print(response)

        image_count += 1

    time.sleep(sleep_time)



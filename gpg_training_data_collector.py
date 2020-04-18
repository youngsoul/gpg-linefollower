import easygopigo3
import signal
import time
import sys
import argparse
from imagezmq.asyncimagesender import AsyncImageSender
from imutils.video import VideoStream
import imutils as imutils
import socket


"""
This script is executed on the GoPiGo.

The purpose of this script is to send images to the remote server and take in driving commands.  

This will allow the GPG to be driven on a training track sending training data back to the server
"""
gpg = easygopigo3.EasyGoPiGo3()
gpg.set_speed(100)
sleep_between_image_sends = 0.25


def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

def setup_async_image_sender():
    # get the host name, initialize the video stream, and allow the
    # camera sensor to warmup
    rpiName = socket.gethostname()

    video_stream = VideoStream(usePiCamera=True).start()

    async_image_sender = AsyncImageSender(server_name=rpiName, server_ip=server_ip, port=5555, send_timeout=10,
                                           recv_timeout=10, show_frame_rate=10)

    return video_stream, async_image_sender

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server-ip", required=False, default='192.168.1.208',
                    help="ip address of the server to which the client will connect")
    ap.add_argument("-r", "--rotate", required=False, default=0, help="Rotate the image by the provided degrees")

    args = vars(ap.parse_args())
    server_ip = args['server_ip']
    rotation = float(args['rotate'])

    video_stream, async_image_sender = setup_async_image_sender()

    while True:
        frame = video_stream.read()
        if frame is not None:
            if rotation != 0:
                frame = imutils.rotate(frame, rotation)

            response = async_image_sender.send_frame_immediate(frame)
            print(f"Command: {response}")
            if response == b'exit':
                break
            elif response == b'stop':
                gpg.stop()
            elif response == b'forward':
                gpg.forward()
            elif response == b'backward':
                gpg.backward()
            elif response == b'right':
                gpg.turn_degrees(20, blocking=True)
                gpg.forward()
            elif response == b'left':
                gpg.turn_degrees(-20, blocking=True)
                gpg.forward()
            elif response == b'straight':
                gpg.turn_degrees(0, blocking=True)
                gpg.forward()
            elif response == b'OK':
                pass
            else:
                print(f"Unknown command: {response}")

        time.sleep(sleep_between_image_sends)

    gpg.reset_all()

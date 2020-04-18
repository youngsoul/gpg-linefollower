import easygopigo3
import signal
import time
import sys


def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

gpg = easygopigo3.EasyGoPiGo3()
gpg.set_speed(80)

print("Press ctrl-c to exit")

signal.signal(signal.SIGINT, signal_handler)

print("driving forward....")
gpg.forward()
time.sleep(2)

gpg.stop()
time.sleep(1)

gpg.backward()
time.sleep(2)

gpg.stop()
time.sleep(1)

gpg.turn_degrees(180, blocking=True)
time.sleep(2.5)

gpg.turn_degrees(-180, blocking=True)
time.sleep(2.5)

print("resetting your GoPiGo3...")
gpg.reset_all()




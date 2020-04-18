import easygopigo3
import signal
import time
import sys


def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

gpg = easygopigo3.EasyGoPiGo3()

print("Press ctrl-c to exit")

signal.signal(signal.SIGINT, signal_handler)

def scenario1():
    gpg.set_speed(80)
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


def scenario2():
    gpg.forward()
    time.sleep(1)
    gpg.stop()
    gpg.turn_degrees(45, blocking=True)
    time.sleep(1)
    gpg.forward()
    time.sleep(1)
    gpg.stop()



if __name__ == '__main__':

    scenario1()

    print("resetting your GoPiGo3...")
    gpg.reset_all()

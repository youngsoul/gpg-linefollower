import easygopigo3
import signal
import sys
import time


WHEEL_SPEED_CONSTANT = 30

gpg = easygopigo3.EasyGoPiGo3()

# (left multiplier, right multiplier)
left_turn = (0.60, 1.40)
right_turn = (1.40, 0.06)

def signal_handler(sig, frame):
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    print("straight")
    gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
    gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
    time.sleep(3)

    while True:

        print("left")
        gpg.set_motor_power(gpg.MOTOR_LEFT,WHEEL_SPEED_CONSTANT*left_turn[0])
        gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT*left_turn[1])
        time.sleep(2)

        print("straight")
        gpg.set_motor_power(gpg.MOTOR_LEFT,WHEEL_SPEED_CONSTANT)
        gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
        time.sleep(3)

        print("right")
        gpg.set_motor_power(gpg.MOTOR_LEFT,WHEEL_SPEED_CONSTANT*right_turn[0])
        gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT*right_turn[1])
        time.sleep(2)

        print("straight")
        gpg.set_motor_power(gpg.MOTOR_LEFT,WHEEL_SPEED_CONSTANT)
        gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
        time.sleep(3)





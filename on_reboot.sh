#!/bin/bash
source "/home/pi/.virtualenvs/gopigo3/bin/activate"
cd /home/pi/dev/linefollower
python drive_by_model.py --server-ip 192.168.1.208 --send-images 1



## ZMQ

`pop install pyzmq`

## Run Image Server

`source ~/.virtualenvs/py36cv4_venv/bin/activate`

`python receive_images.py`

`python server_training_data_collector.py`


## RPI

`source ~/.virtualenvs/gopigo3/bin/activate`

`python ./send_immediate_images.py --server-ip 192.168.1.208`

* You have to train the model on the RPI because you cannot train and save the model on one architecture and load it from another cpu architecture.

* You have to transfer the training images to rpi

* train model and save model

## Train Model

Training the model is handled by a different project

/Users/patrickryan/Development/python/mygithub/gpg3-linefollow-model

This will save a model file that can be loaded
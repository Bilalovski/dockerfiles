import time

import numpy as np
import json
import mxnet as mx
import paho.mqtt.client as paho
from pydust import core
from mxnet.gluon.data.vision import transforms
import pickle
from PIL import Image
import sys
published = False
choice = 1
global img_list
img_list = []


def begin():
    img = img_list.pop()
    param = sys.argv
    choice = 1
    if len(param) > 2:
        choice = param[1]
        model_path = param[2]
        label_path = param[3]
    if len(param) == 1:
        choice = 1
    if choice == 1:
        model_path = "squeezenet1.1-7.onnx"
        label_path = "synset.txt"
        img = mx.ndarray.array(img)
        img = preprocess(img)
        img = mx.ndarray.array(img).asnumpy().tolist()
        data = {'choice': choice, 'data': img, 'onnx_path': model_path, 'label_path': label_path}
        payload = json.dumps(data)
        client.publish("cl_preprocess_out", payload, qos=0)

    while not published:
        pass


def receive(arg):
    img = pickle.loads(arg)
    img_list.append(img)


def on_connect(mqtt_client, obj, flags, rc):
    if rc == 0:
        print("connected")
    else:
        print("connection refused")


def on_publish(client, userdata, mid):
    print("published data")
    global published
    published = True


def preprocess(img):
    transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img


broker = "broker.mqttdashboard.com"
client = paho.Client("cl_preprocessor")
client.on_connect = on_connect
client.on_publish = on_publish
client.connect(broker)
client.loop_start()

dust = core.Core("classify_sub", "./modules")

# start a background thread responsible for tasks that shouls always be running in the same thread
dust.cycle_forever()
# load the core, this includes reading the libraries in the modules directory to check addons and transports are available
dust.setup()
# set the path to the configuration file
dust.set_configuration_file("configuration.json")
# connects all channels
dust.connect()
time.sleep(1)
# add a message listener on the subscribe-tcp channel. The callback function takes a bytes-like object as argument containing the payload of the message
dust.register_listener("classify_image", receive)
# dust.register_listener("subscribe-mqtt", lambda payload: print("Received payload with %d bytes" % len(payload)))

while True:
    if not (len(img_list) == 0):
        begin()
    time.sleep(1)

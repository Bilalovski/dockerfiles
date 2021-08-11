import json
import time

import mxnet as mx
import numpy as np
import paho.mqtt.client as paho
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from collections import namedtuple

global datalist
datalist = []
global data


def begin():
    data = datalist.pop()
    choice = data['choice']
    if choice == 1:
        model_path = data['onnx_path']
        sym, arg_params, aux_params = import_model(model_path)
        Batch = namedtuple('Batch', ['data'])
        if len(mx.test_utils.list_gpus()) == 0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(0)
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

        img = mx.ndarray.array(data['data'])
        mod.forward(Batch([img]))
        # Take softmax to generate probabilities
        scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
        # print the top-5 inferences class
        scores = np.squeeze(scores)

        data = {'choice': 1, 'scores': scores.tolist(), 'label_path': data['label_path']}
        payload = json.dumps(data)
        client.publish("cl_inference_out", payload)


def on_connect(mqtt_client, obj, flags, rc):
    if rc == 0:
        client.subscribe("cl_preprocess_out", qos=0)
        print("connected")
    else:
        print("connection refused")


def on_message(clientName, userdata, message):
    print("message received")
    global data
    data = json.loads(message.payload.decode('utf-8'))
    datalist.append(data)


broker = "broker.mqttdashboard.com"
client = paho.Client("cl_inference_node")
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker)
client.loop_start()

while 1:
    if not len(datalist) == 0:
        begin()

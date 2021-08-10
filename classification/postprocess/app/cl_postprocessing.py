import time
import json
import numpy as np
import paho.mqtt.client as paho
from pydust import core
import pickle
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
def on_message(clientname, userdata, message):
    time.sleep(1)
    data = json.loads(message.payload.decode('utf-8'))
    global choice
    choice = data['choice']
    global score, disp
    if choice == 1:
        scores = np.array(data['scores']).astype('float32')
        a = np.argsort(scores)[::-1]
        for i in a[0:1]:
            print('class=%s ; probability=%f' % (labels[i], scores[i]))
            send_dust(pickle.dumps((labels[i], scores[i])))


def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        client.subscribe("cl_inference_out", qos=0)
        print("connected")
    else:
        print("connection refused")

def send_dust(payload):
    dust = core.Core("classify_pub", "./modules")

    # start a background thread responsible for tasks that shouls always be running in the same thread
    dust.cycle_forever()

    # load the core, this includes reading the libraries in the modules directory to check addons and transports are available
    dust.setup()

    # set the path to the configuration file
    dust.parse_configuration_file("configuration.json")

    # connects all channels
    dust.connect()
    time.sleep(1)
    # declare a bytes-like payload object
    # publishes the payload to the given channel (as defined by the configuration file)
    dust.publish("pub_postProcess", payload)
    time.sleep(1)

    # disconnects all channels and flushes the addon stack and transport.
    dust.disconnect()

    # stops the background thread started by cycleForever() and wait until the thread has finished its tasks before exiting the application
    dust.cycle_stop()

broker = "broker.mqttdashboard.com"
client = paho.Client("cl_postprocessor")
client.on_message=on_message
client.on_connect=on_connect
client.connect(broker)
client.loop_start()
while 1:
    pass

# dockerfiles

These folders contain the dockerfiles for object detection, segmentation and classification.
Each process contains 3 docker files, preprocessing, the inference, and postprocessing. The input that is expected is a raw image stream using DUST, the output given depends on the process, but is always published using dust. Communication between the docker containers has been realized using MQTT.

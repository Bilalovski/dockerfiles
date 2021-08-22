# Overview #
This folder contains the nescessary files for segmentation using onnx models. Configuring the project to include more models will require adding the preprocessing steps along with postprocessing steps (if needed). Every module works using a stack, whenever new data is received it is put in a stack from where we continuously pop data. As such, the last received message will have priority over a earlier received message.
preprocessing and inference steps are the same as the object detection module, aswell as the configuration steps and demo. However the output of the onnx session will not have only bounding boxes but also masks that will enable us to preform segmentation on an image.


## Postprocessing ##
We can postprocess the data according to the choice variable we got. The postprocessor will use the boxes and masks to preform segmentation on the image. The image aswell as the mask is published on a channel using dust.

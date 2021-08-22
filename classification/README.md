# Overview #
This folder contains the nescessary files for classification using onnx models. Configuring the project to include more models will require adding the preprocessing steps along with postprocessing steps (if needed). Every module works using a stack, whenever new data is received it is put in a stack from where we continuously pop data. As such, the last received message will have priority over a earlier received message.
preprocessing and inference steps are the same as the object detection module, aswell as the configuration steps and demo. However the output of the onnx session will not have any bounding boxes, this time every class will have its own score depending on what the onnx model classifies the image as.


## Postprocessing ##
We can postprocess the data according to the choice variable we got. The postprocessor will sort the output of the onnx model from highest score to lowest, the highest scoring class will be published using dust:
``` a = np.argsort(scores)[::-1] ```

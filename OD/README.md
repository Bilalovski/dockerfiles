# Overview #
This folder contains the nescessary files for object detection using onnx models. Currently 2 onnx models are supported: RCNN and Mobinet, configuring the project to include more models will require adding the preprocessing steps along with postprocessing steps (if needed).

## Preprocessing ##
the preprocessing modules reads in data published by dust, this data needs to be an image that has been converted to bytearrays. For demo purposes, we used pickle to convert an image to bytes.
Multiple parameters can be passed along, but they are not required: 
*the choice of the preprocessing function to be used: this is an int, 1 will preform the preprocessing steps for the rcnn model, while 2 will preprocess according to required mobinet input.  
*the model path to be used in the actual inference
*the path for the labels to be used in the postprocessing module*
The parameters must be passed along only in the preprocessing module, they are then automatically sent to the other modules along with other data using mqtt. If no parameters are passed along, then the default choice of model will be rcnn along with its pre- and postprocessing functions.
```data = {'choice': choice, 'data': result.tolist(), 'onnx_path': model_path, 'label_path': label_path} ```
         
The output is a json containing the choice of functions, the model path, the labels path, and finally the actual data, which is the preprocessed image, that will serve as an input for the onnx model





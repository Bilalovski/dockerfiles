# Overview #
This folder contains the nescessary files for object detection using onnx models. Currently 2 onnx models are supported: RCNN and Mobinet, configuring the project to include more models will require adding the preprocessing steps along with postprocessing steps (if needed). Every module works using a stack, whenever new data is received it is put in a stack from where we continuously pop data. As such, the last received message will have priority over a earlier received message.

## Preprocessing ##
the preprocessing modules reads in data published by dust, this data needs to be an image that has been converted to bytearrays. For demo purposes, we used pickle to convert an image to bytes.
Multiple parameters can be passed along, but they are not required: 

*the choice of the preprocessing function to be used: this is an int, 1 will preform the preprocessing steps for the rcnn model, while 2 will preprocess according to required mobinet input.  
*the model path to be used in the actual inference
*the path for the labels to be used in the postprocessing module*

The parameters must be passed along only in the preprocessing module, they are then automatically sent to the other modules along with other data using mqtt. If no parameters are passed along, then the default choice of model will be rcnn along with its pre- and postprocessing functions.

```data = {'choice': choice, 'data': result.tolist(), 'onnx_path': model_path, 'label_path': label_path} ```
         
The output is a json containing the choice of functions, the model path, the labels path, and finally the actual data, which is the preprocessed image, that will serve as an input for the onnx model

## Inference ##

Once the data sent by the preprocessor is received, the input to the onnx files is used for an inference, the onnx model that is used depends on the choice variable that was passed along by the preprocessor using mqtt. The model path is now no longer necessary and can be discarded from the next json payload. Now the output of the onnx model is sent, along with the choice variable and the label path.

## Postprocessing ##
Finally we can postprocess the data according to the choice variable we got. The postprocessor will calculate the classes and their bounding boxes:
``` data.append({str(classes[int(label)]): {"x": x1, "y": y1, "z": 100, "width": x2 - x1, "height": y2 - y1}}) ```

once all the classes above a score of 0.7 have been detected, they are put in an array and sent using dust, the channel is pub_postProcess_OD.


# Configuration


configuring the containers will require multiple steps. If the model that is used cannot have inputs that use a preprocessing function that is already available, then the preprocessing function needs to be added under a new choice, which in our case, was 1 for rcnn and 2 for mobinet, the next one could be 3 for a new model. The preprocessed image will then have to be published using mqtt. Currently the channel used is preprocess_out.

Depending on the amount of outputs given by the onnx model, a new onnx inference function might need to be created. Currently the onnx inference module supports onnx models with 3 or 4 outputs. If the new model has a different amount of outputs, a new onnx inference function will need to be added.

Lastly, if the available postprocessing functions are not compatible with the desired model, then a new postprocessing function will need to be added, once the postprocessing is finished, the data should be sent using dust.


# Demo

To test the containers, all that is needed is building the docker containers in each folder (docker build -t "tag_name" . . Once all of them are build and running, you can publish an rgb image using dust, do make sure the channel names are correct (see configuration.json for channel names). You can then receive the output of the postprocessor using another dust script.

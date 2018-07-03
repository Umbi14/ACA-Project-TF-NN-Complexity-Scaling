# ACA-Project-TF-NN-Complexity-Scaling
Advanced Computer Architecture Project: Tensorflow Neural Network Complexity Scaling

The "config.json" is the file where you can set all the parameters:
- n_layers: number of layers. Each layer has a conv layer and a max pool layer.
- n_filters: number of filters for each layer.
- batch_size
- input_size
- n_classes: the input image can be classified in "n_classes" different classes
- kernel_size
- fc_units
- kernel_stride : kernel stride in the conv layer

Example:
```
{
  "n_layers" : [2,4,5],
  "n_filters" : [16,32,64],
  "batch_size" : [1,2,5,10],
  "input_size" : [[28,28,3],[56,56,3]],
  "n_classes" : [2],
  "kernel_size" : [[3,3],[5,5]],
  "fc_units" : [512],
  "kernel_stride" : [1,2]
}
```

The file "run_inference.py" gets the config file as inputs, generates all the possible configurations and for each one of it
creates a model and runs the inference.

For each configurations it computes the following parameters:
- FLOPS
- Model weights
- Runtime memory

(if you run "run_inferences_mongo.py", it also stores every computed information in mongodb)

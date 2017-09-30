# ExploringTensorFlow
Example model for getting started with TensorFlow using TFLearn using a simple Ham or Spam score model. 

##Overview

This is some of the example code used in my presentation for Exploring TensorFlow. 

This model is far from perfect but could be useful in getting started. You can read a pdf of my slides [here](https://t.co/QvGoWgvtWC) which should help explain whats going on.


The SMS dataset is [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).

## How to build

`python3 train.py`

## Freeze Model

For mobile use you will need a protobuf file. 

Run `python3 freeze.py --model_folder ./model` after you have trained the model. 

This will generate a pb file which can be thrown into the assets folders of an Android app.

## Using TensorBoard
If you have already ran the training script run `tensorboard --logdir /tmp/tflearn_logs/` to see and explore the graph.

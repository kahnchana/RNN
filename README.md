# RNN
RNN will LSTM cells 


I've made inputs and trained on them. The code still has some bug. I get an accuracy of zero. Or perhaps I've messed up when pre-processing the datasets. 

So the datasets (they're a little big) are on my drive. They can be accessed down here.
https://drive.google.com/drive/folders/0B0t3X5WMpC_BWjJIUHV2NUdvVFU?usp=sharing

train.lua is the main file that runs it. The LSTM cell in LSTM.lua is directly taken from the model used in the paper we discussed. lstm_init.lua sets up the lstm according to the dataset size. rand_data.lua imports the data (it also has an option to generate random inputs and outputs). The utils contains code for setting cmd options and printing. 

These links are helpful to understand the basic code blocks used: LSTM as cell, classNLLCriterion as loss function and some simple layers for connecting the inputs to the outputs. 

https://github.com/jcjohnson/torch-rnn/blob/master/doc/modules.md (LSTM)
https://github.com/torch/nn/blob/master/doc/criterion.md (classNLLCriterion)
https://github.com/torch/nn/blob/master/doc/simple.md (view, linear)

## How to run:

1) Download the data : about 120MB
   (four matlab matrices; these are only for training on biking videos; y matrices have 1-negative,2-positive)  
   
2) Install dependencies for torch  

3) Run in terminal: train.lua

4) Run in terminal: test.lua -checkpoint checkpoints/checkpoint_final.t7

You should get the accuracy. But the accuracy keeps coming as zero. I can't figure out whats wrong. 

## Dependencies:

Before running the files, these dependencies must be installed.

luarocks install torch

luarocks install nn

luarocks install optim

luarocks install image


sudo apt-get install libmatio2

luarocks install matio


Simply type in each of the lines above into the terminal in linux to get these installed. 


## DATA Preprocessing

So first all the vectors were turned into size 59X1000. The max time steps were 59. So the others were filled with 0s to make the same size. This has been done in other cases. https://github.com/fchollet/keras/issues/85
This is needed because our batch size is greater than one. If we don't do this, we have to use a batch size of one. I have used normal gradient descent, using the entire batch at once (we have only around a 1000 cases and error falls down fast). 

The dataset had 1531 videos; 1072 are taken for training and 459 for testing. 
The x_train is made of size 1531x59x1000.
The y_train is made of size 1531 (the loss function used require 1-D tensors: this is not supported in matlab, so reshaped on torch).
The x_test is made of size 459x59x1000.
The y_test is made of size 459.

This is fed into the database using a function in rand_data.lua. The data given for this are four matrices of size 1531x59x1000 (x_train), 1531x1 (y_train), 459x59x1000 (x_test) and 459x1 (y_test). These are in .mat format. These should be input to the function getData in rand_data.lua. 


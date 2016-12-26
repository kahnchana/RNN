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

How to run:

1) Download the data : about 120MB
   (four matlab matrices; these are only for training on biking videos; y matrices have 1-negative,2-positive)  
   
2) Install dependencies for torch  

3) Run in terminal: train.lua

4) Run in terminal: test.lua -checkpoint checkpoints/checkpoint_final.t7

You should get the accuracy. But the accuracy keeps coming as zero. I can't figure out whats wrong. 

Dependencies:

Before running the files, these dependencies must be installed.

luarocks install torch

luarocks install nn

luarocks install optim

luarocks install image


sudo apt-get install libmatio2

luarocks install matio


Simply type in each of the lines above into the terminal in linux to get these installed. 

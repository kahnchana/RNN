# RNN
RNN will LSTM cells 


This is just the raw code. I've made some random inputs and outputs which can be used to run the RNN. I haven't tested this on any actual datasets yet. Will do that soon.

train.lua is the main file that runs it. The LSTM cell in LSTM.lua is directly taken from the model used in the paper we discussed. lstm_init.lua sets up the lstm according to the dataset size. rand_data.lua generates random inputs and outputs. 

The utils contains code for setting cmd options and printing. 

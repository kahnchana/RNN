# RNN
RNN will LSTM cells 

This basically takes a set of sequences and classifies these sequences. 

So the datasets (they're a little big) are on my drive. They can be accessed down here.
https://drive.google.com/drive/folders/0B0t3X5WMpC_BWjJIUHV2NUdvVFU?usp=sharing

train.lua is the main file that runs it. The LSTM cell in LSTM.lua is directly taken from the model used in the paper we discussed. lstm_init.lua sets up the lstm according to the dataset size. rand_data.lua imports the data (it also has an option to generate random inputs and outputs). The utils contains code for setting cmd options and printing. 

These links are helpful to understand the basic code blocks used: LSTM as cell, CrossEntropyCriterion as loss function and some simple layers for connecting the inputs to the outputs. 

https://github.com/jcjohnson/torch-rnn/blob/master/doc/modules.md (LSTM)
https://github.com/torch/nn/blob/master/doc/criterion.md (CrossEntropyCriterion)
https://github.com/torch/nn/blob/master/doc/simple.md (view, linear, transpose)

## How to run:

1) Download the data : about 120MB    (two matlab matrices. insert into a folder called data. folder data should be in same folder as the code.) 
2) Install dependencies for torch
3) Run in terminal: train.lua
4) Run in terminal: test.lua -checkpoint checkpoints/checkpoint_final.t7

You should get the accuracy.

## Dependencies:

Before running the files, these dependencies must be installed.

* luarocks install torch
* luarocks install nn
* luarocks install optim
* luarocks install image

* sudo apt-get install libmatio2
* luarocks install matio

Simply type in each of the lines above into the terminal in linux to get these installed. 


## DATA Preprocessing

So first all the vectors were turned into size 59X1000. The max time steps were 59. So the others were filled with 0s to make the same size. This has been done in other cases. https://github.com/fchollet/keras/issues/85
This is needed because our batch size is greater than one. If we don't do this, we have to use a batch size of one. I have used normal gradient descent, using the entire batch at once (we have only around a 1000 cases and error falls down fast). 

The dataset had 1532 videos; training and testing are split 7:3 (N:M)
The x_train is made of size Nx59x1000.
The y_train is made of size N (the loss function used requires 1-D tensors: this is not supported in matlab, so reshaped on torch).
The x_test is made of size Mx59x1000.
The y_test is made of size M.

This is fed into the database using a function in rand_data.lua. The data given for this are two matrices of data and labels. These are in .mat format. These should be input to the function getData in rand_data.lua. (already done in code)

Also, when running the test, the x_test tensor is expanded to size Nx59x1000 (from Mx59x1000). The additional cells can be filled with anything since they're not used. This is done because the RNN model is shaped to take in data of size Nx59x1000. It's easier to just resize this and discard additional data instead of resizing entire RNN (which has to be done after training: so I'm not sure how to do this without affecting the trained weights). 


## Architecture

Initially, 59 time steps were considered. However, 95% of the videos had below 20 time steps. So 20 time steps were considered afterwards, as this was computationally more efficient. 

Also the accuracy on test data showed only a minor decrease of below 0.5% when timesteps beyond 20% are ommited. Therefore, 20 time-steps for inputs were considered.

Also two layers of LSTM units were considered as well. However, in this case, the training would not converge for even an extended about of training cycles (twice the usual). So this was also not considered. 

Thereafter,the architecture of the RNN used was as follows. 

nn.Sequential {

  input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> output
  
  * (1): nn.LSTM(1000 -> 30)
  * (2): nn.Transpose
  * (3): nn.View(-1, 20)
  * (4): nn.Dropout(0.600000)
  * (5): nn.Linear(20 -> 1)
  * (6): nn.View(-1, 30)
  * (7): nn.Linear(30 -> 2)
  
}

Inputs were matrices of size N x 20 x 1000. The LSTM was used to extract 30 features out of the 1x1000 time-varying variables. The output is of size N x 20 x 30. This output is reshaped into two dimensions to apply linear transforms. A dropout layer is used as a regularizor to avoid overfitting of data. 
The first linear layer is used to extract data from the LSTM hidden states across time. The second is used to combine the features extracted from the 30 different LSTM cells. 
CrossEntropyCritereon is used as the loss function during training. 

The architecture was based on the model used for activity recognition in https://arxiv.org/pdf/1411.4389.pdf. Idees were also taken from this models used in https://arxiv.org/pdf/1303.5778v1.pdf and http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf. 


## Experiments

With regards to the YouTube DataSet (11 classes), binary classification was initially carried out separately for each class. Afterwards, multi-class classification was done considering all classes. 

For binary classification, three datasets were used: 28,46 & 82 (28 means 20% motion & 80% static vector components). For each dataset, training was done until model fit training data 99% or better. The accuracies (correct cases percentage) are shown below. Training was done with 20 time-steps for training data. 

| Class         | 28        | 46    | 82    |
| ------------- |:---------:| :----:| :---: |
| biking        | 96.2      | 95.4  | 92.6  |
| diving        | 93.1      | 93.1  | 89.6  |
| golf          | 93.3      | 93.3  | 92.8  |
| juggle        | 94.3      | 93.7  | 90.2  |
| jumping       | 96.5      | 94.1  | 93.1  |
| riding        | 96.1      | 95.7  | 90.2  |
| shooting      | 91.7      | 90.4  | 91.9  |
| spiking       | 94.5      | 93.9  | 93.0  |
| swing         | 94.6      | 94.1  | 91.7  |
| tennis        | 95.9      | 94.1  | 93.3  |
| walk          | 96.1      | 95.7  | 91.9  |

The best accuracies were seen for 28 (20% motion vector and 80% static vector). 

Further testing was carried out using the 28 dataset. Next all time-steps present were used for training (59 time-steps). The accuracies are below. 

| Class         | 28        |
| ------------- |:---------:| 
| biking        | 96.3      | 
| diving        | 92.4      | 
| golf          | 93.0      | 
| juggle        | 95.2      | 
| jumping       | 96.5      | 
| riding        | 96.3      | 
| shooting      | 91.9      | 
| spiking       | 93.2      | 
| swing         | 95.4      | 
| tennis        | 95.9      | 
| walk          | 96.3      | 

Training was also carried out for a variant architecture. 

nn.Sequential {

  input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output
  
  * (1): nn.LSTM(1000 -> 30)
  * (2): nn.Narrow
  * (3): nn.Transpose
  * (4): nn.View(-1, 30)
  * (5): nn.Dropout(0.600000)
  * (6): nn.Linear(30 -> 2)
  
}

The linear layer combining all hidden states of LSTM was omitted, and the final hidden state only was considered. This variant gave similar results to that with the linear layer. However convergence during training took somewhat longer for this model. It did not converge when training some classes. 20 time-steps were considered here.   

| Class         | 28        |
| ------------- |:---------:| 
| biking        | 96.5      | 
| diving        | 93.9      | 
| golf          | 93.4      | 
| juggle        | 92.4      | 
| jumping       | *         | 
| riding        | 97.4      | 
| shooting      | 92.3      | 
| spiking       | 93.4      | 
| swing         | 94.1      | 
| tennis        | 95.8      | 
| walk          | *         |

*did not converge

Finally multi-class training was also carried out. This was done for the 28 dataset. Initally 20 time-steps were considered and training was done. The model was trained until it fit the training set upto 98.75% (convergence stopped at this point). An accuracy of 60.0% was recorded. 
Next, the same was carried out considering 59 time-steps. Training was done until model fit training data 95.42% (convergence stopped afterwards). An accuracy of 62.826 was recorded. 

| Class         | 28        |
| ------------- |:---------:| 
| 20 time-steps | 60.000    | 
| 59 time-steps | 62.826    | 


## References

Code was borrowed from the following libraries. 

* https://github.com/Element-Research/rnn/blob/master/LSTM.lua
* https://github.com/Element-Research/dataload
* https://github.com/jcjohnson/torch-rnn
* https://github.com/garythung/torch-lrcn

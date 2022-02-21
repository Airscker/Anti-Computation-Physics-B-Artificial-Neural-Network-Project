# Anti-Computation-Physics-B-Artificial-Neural-Network-Project 0.22.2

##Attention
This is a responsity for any one who is bothered by ZRY's computation physics B-Neural network homework, not aim to teach you about the true CNN/DNN or SML/SDL knowledge,anyone who use this into business will be claimed and anyone who want to report it as abusement will be claimed by all USTCers, at the same time, this responsity will be turned into secret and unfree.

##Q&A

###How to use?

download the code packet as .zip, then unzip it put all .py files into your project root filefolder, make sure your IDE will find it when you want to use it in your codes.

####what need to do before opening IDE?

1.open powershell/cmd(on windows 10/11),then type ```pip install [the name of package you need to install]```

Tips:   if your shell say that there is no 'pip' command exists, or other errors about pip, just uninstall anacoda/python installed before(Visual studio is the same, just unistall python in control panel in your pc)open [Home page of Python downloading](https://www.python.org/downloads/) and download 3.9.X edition, then when you are installing newly downloaded python, make sure you are agree with 'Add python into system environment' !!!,then after installation everything will be no problem. Please forgive me that anacoda is really a shit,don't use it instead of using VS/VS code,pycharm is a second choice,just because it's not free for professional edtion.

2.now the packages you need to install:

cmake
boost
matplotlib
keras
matplotlib
numpy
scipy
pandas
openpyxl
Scrapy
sklearn

3.which edition of Tensorflow you need to install?

There is no question that you need to install tf 2.x,but here I mean that GPU or CPU edtion you need to install?

First you need to know the Independent GPU's name, such as my RTX 3070,or your MX/GTX series GPUs
Then you need to find your GPUs compute capability on the NVIDIA/AMD(Depend on the maker of your GPU) home page,if your GPU's capability is below 5.0,you just install CPU edtion(neural network will be trained on CPU):
```pip install tensorflow```

If not, you need to know that GPU's train speed is a Exp function, which means that it's slower when your train loops is too small.If you really want to use GPU, let NVIDIA as a example, first download CUDA and CUDDN and install it, then:
```pip install tensorflow-gpu```

####Start a engine

when you want to train a neural network, just use import key word in your code file(.py), just do it as this way:

```python
N1=KERAS_NeuralNet()
N1.KERASNeuralEngine()
```

Tips:   Do not think that it's out of the command of teacher when you use keras neural network frame, actually you need to konw that Tensorflow 2.X edtion uses almost whole keras system codes and engines, that is to say, you can view Tensorflow 2.X edition as the same as keras, if ZRY/WYS or anybody query your codes, nothing will be clarified except for the truth that he/she is a f*** asshead.

####Parameters of KERAS_NeuralNet()

dataset:the 2-dimension list of all data(containing train/test data),such as [[name1,age1,sex1],[name2,age2,sex2]...]
labels:the 1-dimension list of data labels
init_lr:learn rate(0.1)
Data_type:the type of your dataset's data(TEXT and IMG is not supported in such simple network)(NUM)
Batch_size:the size of your batch(32)
epochs:how much loops you want to train(10)
div_prop:the proportion of the test dataset in whole dataset(0.1)
Dense_num:neural network construction,such as [2,128,2] means that 2-128-2 neural network planes
opti:Optimizer
plot_al:Plot the graph of error and accuracy(True/False)

####Return value of KERASNeuralEngine()

The accuracy of the model in your test dataset.

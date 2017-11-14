# COMP551_A3

This repo is for a project which looks to classify handwritten digits and characters. The dataset is available at https://www.kaggle.com/c/comp551-modified-mnist/data. The dataset contains 50,000 64x64 grey scale images which fall into one of the 40 classes: ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]). Each image has two numbers (between 0 and 9) and one letter (A/a or M/m). If A is detected the digits are added, and if m is detected the digits are multiplied. 

This repo contains 4 different algorithms to classify these images: Logistic regression, Feed-foward neural net, and two differnt CNN architecture. One CNN architecture alternates CNN and max pooling layers and then one implements a GoogLeNet structure. To run any of the algorithms in repo follow the directions.

First download the files from https://www.kaggle.com/c/comp551-modified-mnist/data.

Then you need to "pickle" the download files so they can be read by the algorithm (Trust me it saves lots of time if you want to run more than once). To pickle enter in the command line prompt 
    
    python pickle.py

Now once the files are pickle it is necessary to install three libraries: numpy, sklearn, and tensorflow. Both numpy and sklearn can easily be installed using pip. For tensorflow we recommend looking up how to download the correct version for your operating system and cpu/gpu which is avaiable in the tensorflow documentation

Finally run the following commands to execute the programs

1)Logistic Regression
      
    python regression.py
  
2)Feedforward Neural Net
  
    python ff.py
  
3)CNN: 4 conv/pool layers
  
    python CNN1.py
   
4)GoogLeNet
      
    python gnet.py
      
Additionally we have included a script with preprocessing options (preproess.py). In all of the above algorithms we include the preprocessing option that generated the best results for us (binary dilation with a 0.7 threshold), but feel free to load any of those from the avaiable package. Note that most of those preprocessing functions require a (-1,64,64) numpy array as input. The pickled files will save all of your data in the correct format for running in the algorithms as well as the preprocessing 

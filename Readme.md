# Readme #

**JefiAtten** is a novel neural network model employing an attention mechanism to solve **Maxwell's equations** efficiently. JefiAtten uses self-attention and cross-attention modules to understand the interplay between **charge density**, **current density**, and **electromagnetic fields**.

## Acquisition of the Training Set ##

Due to the substantial size of our training dataset, uploading the complete data would necessitate an extensive amount of time. As an alternative solution, we have opted to upload the code required to generate the training set. Here, we will provide a simple description of the code’s operation. Those interested in the underlying principles of the code can refer to Article **[JefiGPU](https://www.sciencedirect.com/science/article/pii/S0010465522000467)** for more information.

Structure of JefiGPU Program:

The file structure within the JefiGPU folder is presented as follows:

- JefiGPU
	- EMsolver
	- 0-251.ipynb
	- 252-499.ipynb
	- data get.ipynb

Within this structure, the EMsolver folder contains the core programs of JefiGPU. These have been encapsulated for ease of use, so an understanding of their internal architecture is not necessary for the process of acquiring the training set.

**0-251.ipynb ，252-499.ipynb** contain 500 sets of predefined equations for current density and charge density. If you wish to replicate our work, you can utilize these initial conditions to generate our dataset.

**data get.ipynb** mainly involves the processes of dataset generation and format conversion. You only need to understand the configuration of certain parameters to swiftly produce the dataset you require.

- Begin by defining the input equations for current density and charge density in **generate rho J**;
- Next, establish the number of grid numbers and their size using **x grid size, dx**;
- Then, determine the time step and the total number of iterations with **dt, lentimesnapshots**;

After setting these parameters, all you need to do is define the location for data storage to proceed with dataset generation. As you will observe, our program conducts three main operations on the dataset: the first creation of the dataset; the second involves splitting the dataset into (current density, charge density) and (electromagnetic fields); the third step involves appending temporal information to the data.

## The structure of JefiAtten ##

To facilitate testing and program adjustments, we have placed JefiAtten in a single Jupyter notebook file (JefiAtten.ipynb).

- Firstly, import the necessary packages, ensuring that all the required packages for JefiAtten are downloaded;
- Secondly, define the number and IDs of GPUs to be used;
- Thirdly, specify the location of your input data within **Local-attention**;
- Fourthly, in **MyDataset**, **MyDataset1**, define the locations where the pre-training data and the formal training data reside;
- Finally, define the parameters required for training:
	- Use **batch size** to specify the number of data points inputted during each training step;
	- **query EM dim, mem source dim, time dim** define the dimensions of the dataset inputs during training (for example, the current density and charge density dimensions are 4, electromagnetic field density is 6, and the time dimension is 10);
	- **embedded dim, output EM dim** determine the dimensions of the network mask and output, respectively;
	- **num heads, num layers** specify the number of heads in the multi-head attention mechanism and the number of attention layers used in the model;
	- **dropout** sets the number of neurons to be dropped in the network;
	- **num gpus, num workers** define the number of GPUs used for data reading and the number of data points pre-choose, respectively;
	- **numepochs, numepochs1, learningrate G** represent the number of pre-training epochs, formal training epochs, and the learning rate, respectively.

After defining the above parameters, you can proceed with the training of the JefiAtten model, and the learned model is model_S. If you wish to use the JefiAtten model directly, we also provide a pre-trained model (**JefiAtten.pth**).

You can use our trained model with the following commands:

modelG = torch.load('20231113200_epochs.pth')

model_G.eval();

This will allow you to utilize the pre-trained JefiAtten model for your purposes.

## License ##

The package is coded by Jun-Jie Zhang and Ming-yan Sun.

This package is free you can redistribute it and/or modify it under the terms of the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

For further questions and technical issues, please contact us at

zjacob@mail.ustc.edu.cn (Jun-Jie Zhang 张俊杰)
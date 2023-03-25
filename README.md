# convolutional_neural_network.py

	- ❗ Disclaimer: ❗

	- This code may bear resemblance to other existing implementations, but it was written by me and intended for educational purposes only. 
	- I make no claim to be the original author of the concepts or methods implemented in this code. 
	- If you believe that any part of this code infringes upon your intellectual property rights, please contact me immediately and I will take appropriate action. 



This project is a Convolutional Neural Network (CNN) designed for the CIFAR-10 image classification task. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The objective of this neural network is to correctly classify these images into their respective categories.
Dependencies

The CIFAR-10 dataset is loaded using the "keras.datasets.cifar10" module from Keras. The dataset is split into training and testing sets, with 50,000 images in the training set and 10,000 images in the testing set. The images are preprocessed by scaling the pixel values between 0 and 1.
Model Architecture

The neural network architecture is defined using the Keras Sequential API. It consists of three convolutional layers with ReLU activation, followed by two max pooling layers. The output of the last convolutional layer is flattened and passed through two fully connected layers with ReLU activation, followed by an output layer with 10 units (one for each class) and no activation function. The architecture can be visualized as follows:

	Input (32x32x3)
	Conv2D (32 filters, 3x3, ReLU)
	MaxPooling2D (2x2)
	Conv2D (64 filters, 3x3, ReLU)
	MaxPooling2D (2x2)
	Conv2D (64 filters, 3x3, ReLU)
	Flatten
	Dense (64 units, ReLU)
	Dense (10 units, no activation)
	
The neural network is trained using the "fit" method of the Keras Sequential model. The training is done for 10 epochs with a batch size of 64. The Adam optimizer is used with the Sparse Categorical Crossentropy loss function and accuracy metric.

The trained model is evaluated on the testing set using the evaluate method of the Keras Sequential model. The accuracy score on the testing set is reported.

To use this neural network for CIFAR-10 image classification, you can simply clone this repository and run the "convolutional_neural_network.py" file. The neural network will be trained and the accuracy on the testing set will be printed. You can also modify the hyperparameters, such as the number of epochs, batch size, and learning rate, to see if you can improve the accuracy score.

    - The "tensorflow" library is developed and maintained by Google Brain team and many contributors from the open source community. 
    - The project is led by Martin Wicke, Rajat Monga, and Joshua Gordon.

Team Members:
	- Antonio Chedid
	- Paul El Kouba


Selected Topic:
	- Emotion Recognition from Visual Media

	
Contributions of team members:
	- In order to choose the topic and the different journal articles with codes, we both worked equally: we both researched and read papers and searched for their relative codes. After choosing our top three papers, we also discussed choosing the best of them. (Paul: 50% ; Antonio 50%)
	- For commenting the original code, getting and using the datasets, and trying to run the Gabor Convolutional Layers, Paul worked more on it than Antonio since Antonio was focusing on extedning the code. (Paul 60% ; Antonio 40%)

	
Publication & Code citations:
	-Publication:
	Jiang, P., Wan, B., Wang, Q., &amp; Wu, J. (2020). Fast and efficient facial expression recognition using a Gabor Convolutional Network. 
			IEEE Signal Processing Letters, 27, 1954–1958. https://doi.org/10.1109/lsp.2020.3031504 
			
	-Code:
	general515. (n.d.). General515/FACIAL_EXPRESSION_RECOGNITION_USING_GCN. GitHub. Retrieved December 6, 2022, 
			from https://github.com/general515/Facial_Expression_Recognition_Using_GCN 
			
			
Datasets:

Multiple datasets were used for training and testing notably the FER2013 and the RAF-DB datasets.
We tried to get the RAF-DB first since it is the one used in the model without the GCN, but we could not download it since we needed access from 
the owners and despite us sending a request as needed, we did not receive an answer with the passkey and the URL to download it.
Thus, we stuck with the FER2013 even though it returned a lower accuracy during testing than what was expected.

	-FER2013:
	https://www.kaggle.com/datasets/msambare/fer2013
	
	-RAF-DB:
	http://www.whdeng.cn/raf/model1.html
	
	
Subfolders:
	Original Code
	|
	|
	|___Model with GCN: This is the model that runs using the Gabor Convolutional Networks. Since we were unable to download it, we did not use it.
		|
		|
		|_____trained_weights: Contains multiple pre-trained models on different datasets.
	|
	|
	|___Model without GCN: This is the model we used and it uses normal Convolutional Neural Networks.
		|				   It contains a pre-trained model, two test images, four .py files to run the testing of the model 
		|				   (on single images, and on datasets)
		|
		|
		|_____FER2013 Dataset: This folder contains the training and testing dataset.
			|
			|
			|_______Train: This folder is not used in our project
			|
			|
			|_______Test:  This folder is used to test the accuracy of the model over a certain dataset
		|
		|
		|_____Images: This folder contains an image called '4 people' and multiple other images that are cropped and derived from this one to 			test the limitations of the model.
	
	
Running the Code:
For running the code, we tried at first to run the one with the GCN. However, despite multiple trials and despite following the instructions provided by
the developers of this model, we failed to launch it on our devices. That is why, we focused on running the pre-trained model without the Gabor networks.
To run the tests and run the code, we have two options:

1) Test the model on a single image:
	To do so, change the directory of the test_image in the code where it is commented "#Change Directory" and run the code.
	It should take around 10 seconds to provide us with an output.
	For our test, we commented out the multiple images used and left only one uncommented.

2)Test the model on a certain dataset:
	To do so, change the directory of the test_image in the code where it is commented "#Change Directory" and run the code.
	Running on a CPU, it should take more than half an hour using the FER2013 test dataset. Since transforming the images take time.
	Running on a GPU, it should take half the time needed for it to run on the CPU.
	
Input Folders: Images & FER2013 Dataset
Output: printed on the terminal.
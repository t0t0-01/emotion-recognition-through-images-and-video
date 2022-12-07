Team Members:
	- Antonio Chedid
	- Paul El Kouba


Selected Topic:
	- Emotion Recognition from Visual Media

	
Contributions of team members:
	-For the transfer learning application and addition of new features, Antonio worked more on this part than Paul, as Paul focused on the original 
	 code and reproducing the results (60% Antonio, 40% Paul).
	
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
	RevisedCode
	|
	|
	|___images: This folder contains images displayed in the IPython (Jupyter) Notebook
	|
	|
	|___Inputs: This folder contains input test data. It contains 3 image files, and one video file.
	|
	|
	|___Outputs: This folder contains the outputs of the code. As of submission, it has 2 videos: results, and results_improved.
	|
	|
	|___haarcascade_frontalface_default.xml: This file contains the trained weights of the Haar Cascades used in the code.
	|
	|
	|___net_factory.py: This file is a dependency file provided by the authors in their original paper
	|
	|
	|___utils.py: This file is a dependency file provided by the authors in their original paper
	|
	|
	|___trained_RAF_10.pt: This file contains the weights for the model that was used in the code.
	|
	|
	|___main.py: This file contains the main revised code. It has the final versions of the functions used; for details of their implementation, refer to the Analysis.ipynb file.
	|
	|
	|___Analysis.ipynb: This is a Jupyter Notebook that contains the detailed analysis steps we followed to reach the final results.

Improvements added:
	To improve the model, we decided to extend on the previously done one and extend on it to create a model that works not only on static images, but on offline and real-time videos. 
		To do so, we created a pre-processing step since there was none in the codes provided by the authors. This pre-processing step consists of detecting faces in the frame to be inputted to the model. 

	We also allowed the detection of multiple faces not only one. In doing so, we surpassed two of the limitations found in the original model.


	Then we thought of a way to integrate memorization and neighborhood analysis to detect the emotions accurately and remove any noise and randomness that the model might face.

Motivation for improvements:
	In their original pipeline, the authors did not include an extensive pre-processing step. The model accepts as input strict type of images and strict conditions for the faces in them. This is not applicable in the real world

	The authors' model works only on one face. However, in the real world, and when analyzing videos in real-time, it is very plausible that several faces appear in the same frame. Thus, the model should accomodate for that.

	During experimentation, we saw that results fluctuated considerably, as the model changed predictions quickly without any change in the facial expression of the user.
	As such, we wanted to include a features that irons out these uncertainties by checking the neighborhood of the frames being analyzed.



Running the Code:
There are two ways to run the code:
	1. Using the main.py file:
		a. Uncomment the model definition
		b. For offline video analysis, uncomment the face_in_vid line
			Include as a first parameter the path to the input video
			Include as a second parameter the path the output video
			Include as a third parameter the model
		c. For real time video analysis, uncomment the real_time_analysis_improved function
	2. Using the Analysis.ipynb file: detailed steps and analysis are included in the notebook itself. The user can follow the instructions provided in the notebook.
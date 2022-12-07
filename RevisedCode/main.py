import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from net_factory import GCN_mod
import numpy as np
from PIL import Image
import time
from scipy import stats
import itertools


def load_model(model_name):
    """


    Parameters
    ----------
    model_name : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """

    model = GCN_mod(channel=4, lych=10)
    model.load_state_dict(torch.load(
        model_name, map_location=torch.device('cpu')))
    model.eval()

    return model


def get_prediction(model, img):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # Preprocess the cropped image. This is model-specific
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(size=(100, 100)),
        transforms.TenCrop(90),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
    ])

    # Apply transformations on the image
    timg = transform(img)

    # Forward pass to get prediction
    with torch.no_grad():
        score = model(timg)

    score = score.mean(dim=0)
    # Additional Softmax Layer
    layer = torch.nn.Softmax(0)
    probabilities = layer(score)

    # Get the output emotion
    outs = ['Anger', 'Disgust', 'Fear', 'Happy',
            'Neutral', 'Sadness', 'Surprise']
    emotion = torch.argmax(probabilities)

    return outs[emotion.numpy()], probabilities


def get_faces_and_prediction(frame, model):
    """
    Returns a dictionary with the location and the label of each face in a frame

    Parameters
    ----------
    frame : PIL Image
        An image.

    model: Pytorch Model
        A model from which to predict the emotion of a face.

    Returns
    -------
    results : Dictionary
        Dictionary, where the keys are the coordinates of the faces, and the
        values are a tuple with the label and probabilities for each face.

    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    results = {}
    # Draw the rectangle around each face labeled with the emotion
    for (x, y, w, h) in faces:
        #Crop the face by itself
        cropped = frame[y:y + h, x:x + h]

        #Convert to PIL image for processing
        cropped_pil = Image.fromarray(np.uint8(cropped))

        #Get a prediction on the emotion
        label, probabilities = get_prediction(model, cropped_pil)

        #Append label of current face to labels
        results[(x, y, w, h)] = label

    return results




def average_of_images(images, model, previous_images):
    """
    Predicts the emotion of faces in an image. This is done by checking the entire neighborhood (nb of neighbors = 3)
    and assigning for a face the emotion that occurred most in this neighborhood.

    Parameters
    ----------
    images : dictionary
        Dictionary where the keys are the frame number in the overall video, 
        and the values are the images themselves.
        NB: The objective of the model is to get the predictions of the middle
        frame.
    model : Pytorch Object
        Model with which to analyze the values.
    previous_images : dictionary
        Dictionary where the keys are the frame numbers of images previously 
        analyzed, and the values are other dictionaries with keys being 
        coordinates of face and values being labels for each face.
        

    Returns
    -------
    averaged_out : dictionary
        Dictionary where the keys are the coordinates of the faces in the image
        that is meant to be processed (the one in the middle of the neighborhood),
        and the values are the labels that occured the most for each face.

    """
    
    #Dictionary that will contain the neighboring frames and the labels of the faces in the frames
    overall = {}
    
    #Take every frame in the neighborhood
    for frame_number in images:
        #If this frame was processed before (i.e., present in the previous_images dic), add it directly
        if frame_number in previous_images:
            overall[frame_number] = previous_images[frame_number]
            
        #If not, run the frame through the model to get the faces and their predictions
        else:
            current_frame_image = images[frame_number]
            faces_and_preds = get_faces_and_prediction(current_frame_image, model)
            overall[frame_number] = faces_and_preds
            previous_images[frame_number] = faces_and_preds
            
    averaged_out = {}
    
    #Get target frame number. To do so, we get the median of the keys, which
        #correspond to the frame numbers. We can do so because the assumption 
        #is that neighborhood consists of an odd number of frames, in 
        #consecutive, increasing order (e.g., [1,2,3], [1,2,3,4,5])
    target_frame_number = int(np.median(list(overall.keys())))
    
    #Get the current faces
    current_keys = overall[target_frame_number].items()
    
    #Define threshold to check if same face
    DIFFERENCE_IN_THRESHOLD = 10
    
    #Remove the target frame from the overall object to process labels below
    overall.pop(target_frame_number)

    for img in overall.values():
        #Get list of possible combinations between faces of the main image and faces of the secondary image
        current_combs = list(itertools.product(current_keys, list(img.items())))
        
        #Check which faces of main correspond to the faces of the secondary
        #Unpack the current_combs into coordinates and labels of main and secondary
        for (main_coordinate, main_label), (secondary_coordinate, secondary_label) in current_combs:
            
            #Variable to keep track if the secondary should be accepted
            accepted = True
            
            #Take the dimensions of the two faces in pair to check if their difference falls within the threshold
            for main_dim, secondary_dim in zip(main_coordinate, secondary_coordinate):
                
                #Check if difference falls within threshold
                if np.abs(main_dim - secondary_dim) > DIFFERENCE_IN_THRESHOLD:
                    #If it does not, reject the current combination and disregard other dimensions
                    accepted = False
                    break
                    
            #If no dimensions were rejected, proceed to accept the combination
            if accepted:
                
                #If face already exists in overall result, just append the new label to the existing list of labels
                if main_coordinate in averaged_out:
                    averaged_out[main_coordinate].append(secondary_label)
                    
                #If face does not exist, create a new entry in the final result, where the values are the main label and the new label
                else:
                    averaged_out[main_coordinate] = [main_label, secondary_label]
    
    #Choose the prediction that appeared most in the neighborhood
    averaged_out_common = {k: str(stats.mode(v).mode[0]) for k, v in averaged_out.items() }
    return averaged_out_common         



def face_in_vid(input_path, output_path, model):
    """
    Processes an input video file by detecting faces and their emotions, and generates a new video file with this analysis

    Parameters
    ----------
    input_path : String
        A path that contains the video that needs to be processed.
    output_path : String
        The path where the resulting video should be in.
    model : Object
        A Pytorch model that will process the video.

    Returns
    -------
    Dictionary
        A dictionary that contains time data of the analysis. 'Overall' key corresponds to the total processing time, and
        'Frames' key corresponds to processing time of each and every frame.

    """
    

    #Read video
    video = cv2.VideoCapture(input_path)

    #Get video dimensions
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    
    #Get number of frames in video
    FRAMES_IN_VIDEO = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    
    #Create output vide (AVI Codec)
    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        
    #Start recording time of entire pipeline
    overall_start = time.time()

    #List of elapsed time for frame-by-frame processing
    frame_dur = []
    
    #Dictionary that contains the list of frames already processed
    images_processed = {}
    
    frame_nb = 0
    while frame_nb != FRAMES_IN_VIDEO:
        #Record time frame by frame
        frame_start = time.time()

        #Increment frame number
        frame_nb += 1
        
        frames_to_process = {}
        
        #Read the frame
        video.set(1, frame_nb)
        ret, img = video.read()
        frames_to_process[frame_nb] = img
        
        #If frame is not first frame, add the previous to the neighborhood
        if frame_nb != 1:
            video.set(1, frame_nb-1)
            ret_prev, frame_prev = video.read()
            frames_to_process[frame_nb-1] = frame_prev
        
        #If frame is not last frame, add the next frame to the neighborhood
        if frame_nb != FRAMES_IN_VIDEO:
            video.set(1, frame_nb+1)
            ret_next, frame_next = video.read()
            frames_to_process[frame_nb+1] = frame_next
            
        
        faces_and_preds = average_of_images(frames_to_process, model, images_processed)
        for (x,y,w,h), label in faces_and_preds.items():
            #Plot the bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #Write the prediction above box
            cv2.putText(img,
                        text=label,
                        org=(x, y - h // 15),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2)

            
        #Append time taken for this frame to overall list
        frame_dur.append(time.time() - frame_start)
        
        
        #Output updated frame to video file
        result.write(img)


    # Release the VideoCapture object
    video.release()
    result.release()

    #Build the history object that contains the time information
    hist = {"Overall": time.time() - overall_start, "Frames": frame_dur}

    return hist


def real_time_analysis_old(model):
    video = cv2.VideoCapture(0)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    outs = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
    prediction_values = {x: [] for x in outs}
    time_values = []
    
    #Get FPS
    FPS = int(video.get(cv2.CAP_PROP_FPS))
        
    frame_nb = 0
    while True:
        #Read the frame
        ret, img = video.read()

        #Condition to check if frame is available
        if ret:
            frame_nb += 1
            #Flip image because camera input is flipped
            img = cv2.flip(img, 1)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.5, 4)

            # Draw the rectangle around each face labeled with the emotion
            for (x, y, w, h) in faces:
                #Crop the face by itself
                cropped = img[y:y + h, x:x + h]

                #Convert to PIL image for processing
                cropped_pil = Image.fromarray(np.uint8(cropped))

                #Get a prediction on the emotion
                label, probabilities = get_prediction(model, cropped_pil)

                for pos, proba in enumerate(probabilities):
                    prediction_values[outs[pos]].append(proba)
                time_values.append(frame_nb / FPS)

                #Plot the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                #Write the prediction on the box
                cv2.putText(img,
                            text=label,
                            org=(x, y - h // 15),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2)
                
                
            # Display the resulting frame
            cv2.imshow('frame', img)
              
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # Release the VideoCapture object
    video.release()
    cv2.destroyAllWindows()

    return prediction_values, time_values


def real_time_analysis_improved(model):
    video = cv2.VideoCapture(0)

    images_processed = {}
    frame_nb = 0
    while True:
        #Increment frame number
        frame_nb += 1
        
        frames_to_process = {}

        #Read the frame
        video.set(1, frame_nb)
        ret, img = video.read()
        frames_to_process[frame_nb] = img
        
        #If frame is not first frame, add the previous to the neighborhood
        if frame_nb != 1:
            video.set(1, frame_nb-1)
            ret_prev, frame_prev = video.read()
            frames_to_process[frame_nb-1] = frame_prev
        
        #Note that there is no concept of last frame here
        video.set(1, frame_nb+1)
        ret_next, frame_next = video.read()
        frames_to_process[frame_nb+1] = frame_next
            
        
        faces_and_preds = average_of_images(frames_to_process, model, images_processed)
        for (x,y,w,h), label in faces_and_preds.items():
            #Plot the bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #Write the prediction above box
            cv2.putText(img,
                        text=label,
                        org=(x, y - h // 15),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2)
    

        # Display the resulting frame
        cv2.imshow('frame', img)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the VideoCapture object
    video.release()
    cv2.destroyAllWindows()

    return ""



model = load_model('trained_RAF_10.pt')
#face_in_vid("./Inputs/test_vid3.mp4", "./Outputs/result.avi", model)
real_time_analysis_improved(model)


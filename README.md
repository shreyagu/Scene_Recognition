# Scene Recognition

Scene recognition plays an important role in a fully developed AI's functionality. Here I present a way of identifying if an image scene is indoor or outdoor. This can be further extended to real-time video scene recognition as well. The method relies on a carefully designed Convolutional Neural Network.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
The following python packages are required for running the software.

[numpy](http://www.numpy.org/)

[OpenCV 3.4.1](https://github.com/opencv/opencv)

[Keras](https://keras.io/)

[Scikit-learn](http://scikit-learn.org/stable/documentation.html)

[Matplotlib](https://matplotlib.org/contents.html)

### Preparing data
The model was trained on images extracted from the frames extracted from YouTube 8M dataset videos. The frames were extracted based on the code [here](https://github.com/gsssrao/youtube-8m-videos-frames). 

A trained model is already saved in 'data' folder. However re-training is supported. Follow below steps to prepare data for training:
1. Download all indoor and outdoor videos and store them separately.
2. Run the extract_frames.py script on each folder with appropriate category as arguement. After this you will have 'Frames' folder for both indoor and outdoor
3. Place the indoor frames to "data/train/Indoor_data" and outfoor frames to "data/train/Outdoor_data"


### Training
To train a new model, prepare the data as mentioned above and run the training.py.

The current model was designed in two steps. Initially below setup was used to train the images.
![alt text](https://raw.githubusercontent.com/shreyagu/Scene_Recognition/master/data/initial_nw.png)
Total training time: 10 hours
Below accuracy and loss plots were obtained for training vs validation.
![alt text](https://raw.githubusercontent.com/shreyagu/Scene_Recognition/master/data/initial_nw_plot.png)

From the plots it is clear that there is a high over-fitting. So a new design was created with an extra layer of drop-outs and max pooling layer to compensate for the high over-fitting. Below is the summary of the model used.
![alt text](https://raw.githubusercontent.com/shreyagu/Scene_Recognition/master/data/final_nw.png)
Total training time: 13 hours
Below accuracy and loss plots were obtained for training vs validation.
![alt text](https://raw.githubusercontent.com/shreyagu/Scene_Recognition/master/data/final_nw_plot.png)


## Formats

### Input format
The inputs need to have the file name and test image name passed as the input arguments

```
python scene_recognition.py test_image.jpg
```

### Output format
The command line will output the predicted category for the scene as **Indoor** or **Outdoor** in the following format.
```
Predicted Category is: Indoor
```

## Sample unit testcase
```
$ python scene_recognition.py santa_monica.jpg
Predicted Category is: Outdoor
```

## Built With
[Jupyter-notebook](http://jupyter.org/) - A web-based notebook environment for interactive computing.

[Anaconda Python Cloud](https://anaconda.org/anaconda/python) - A free and open source distribution of the Python and R programming languages for data science and machine learning related applications.

## References
The YouTube 8M dataset can be downloaded [here](https://research.google.com/youtube8m/download.html). 
Thanks to Pex for providing this challenge.
Also, thanks to all the authors of videos used to perform the study. The dataset shall not be used for any commercial purposes.

import sys
import time
import cv2 as cv2
import numpy as np
from keras.models import load_model


xtest = []

# Read the test image
test_img = sys.argv[1]
test_img = cv2.imread(test_img)

# Reducing the size of image to 64x64x3
rsize = 64
test_img = cv2.resize(test_img,(rsize,rsize), interpolation = cv2.INTER_CUBIC)
xtest.append(test_img)
xtest = np.array(xtest)

# Load the pre-trained model
model = load_model('data/InOutClassifierModel.h5')

# Predict the classification of the test image.
tic=time.time()
ypred = (model.predict(xtest))
toc=time.time()
print(toc-tic)
# Display the result
if (((ypred[0])[1])==1.0):
    print ("Predicted Category is: Outdoor")
elif(((ypred[0])[0])==1.0):
    print ("Predicted Category is: Indoor")

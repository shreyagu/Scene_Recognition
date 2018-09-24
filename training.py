import numpy as np
import cv2 as cv2
import pickle
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model


# Set the constants
batch_size = 128
num_classes = 2
epochs = 200

# Image dimensions for resizing
rsize = 64 


# Plot accuracy and loss plots for training vs validation 
def draw_plots(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Create empty lists for image and class
x = []
y = []

# Read training images and resizing them to 64x64
for i in range(1,3201):
    image_name = str("data/train/Indoor_data/indoor"+ str(i)+".jpg")
    img = cv2.imread(image_name)
    img = cv2.resize(img,(rsize,rsize), interpolation = cv2.INTER_CUBIC)
    x.append(img)
    y.append('0') # 0 reprsenting indoor
    image_name2 = str("data/train/Outdoor_data/outdoor"+ str(i)+".jpg")
    img2 = cv2.imread(image_name2)
    img2 = cv2.resize(img2,(rsize,rsize), interpolation = cv2.INTER_CUBIC)
    x.append(img2)
    y.append('1') #1 representing outdoor
    

# Convert to numpy array
x = np.array(x)
y = np.array(y)


# dividing data into test and train with default np.random shuffle
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.15, random_state = None)

# shuffling data randomly 
idx = np.random.permutation(len(xtrain))
xtrain,ytrain = xtrain[idx], ytrain[idx]

idx = np.random.permutation(len(xtest))
xtest,ytest = xtest[idx], ytest[idx]

# Normalizing each pixel intensity to lie between [0,1]
xtrain = np.array(xtrain,dtype="float")/255.0
xtest = np.array(xtest,dtype="float")/255.0
ytrain = np.array(ytrain)
ytest = np.array(ytest)

print('xtrain shape:', xtrain.shape)
print(xtrain.shape[0], 'train samples')
print(xtest.shape[0], 'test samples')

# Input shape indicator to CNN model
input_shape = xtrain[0].shape


# Convert class vectors to binary class matrices with one-hot encoding
ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)


# Creating the CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Split the training data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.10)

# Train the model on the training data and test each epoch on the validation data
history = model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xval, yval))


# Save the model and history.
model.save('InOutClassifierModel.h5')
pickle.dump(history.history, open("InOutClassifierHistory.bin", 'wb'))


# Evaluate the model on test data
score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot the accuracy and loss plots
draw_plots(history.history)

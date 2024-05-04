import os.path
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt
from jes4py import *
import cvzone
import cv2
from ultralytics import YOLO


# - Loading data
data_dir = 'fruits/dataset/train'
test_dir = 'fruits/dataset/test'

# Create an ImageDataGenerator and rescale the images from 0 to 1 instead of 0 to 255
datagen = ImageDataGenerator(rescale=1./255)
# Use the generator for training data
train = datagen.flow_from_directory(data_dir, target_size=(100, 100), batch_size=32,
                                    class_mode='categorical')
# Use the generator for validation data
val = datagen.flow_from_directory(test_dir, target_size=(100, 100), batch_size=32,
                                  class_mode='categorical')

# - The Model

# Build the model
model = Sequential()
# Adding convolution
# 16 is num of filters, size of filter 3x3 pixels, move 1 pixel each time
# 'relu' activation is converting any negative num to 0 to only preserve positive numbers
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(100, 100, 3)))
# Here we take maximum data after 'relu' activation, it takes 2x2 data and choose max value of them
# to reduce the num of data
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
# Condensing the data into single dimension
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='softmax'))
# We used 'adam' optimizer, the loss we used is for multi classification
# The metric we want to track is accuracy (how well our model is classifying)
model.compile('adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()
# Training
history = model.fit(train, epochs=10, validation_data=val)
# visualize the model accuracy
fig = plt.Figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

def predictedFruitAndState(image_path):
    test_image = image.load_img(image_path, target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    test_image/=255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    predictedClassIndex = np.argmax(result)
    if predictedClassIndex == 0:
        return "Fresh Apple"
    elif predictedClassIndex == 1:
        return "Fresh Banana"
    elif predictedClassIndex == 2:
        return "Fresh Orange"
    elif predictedClassIndex == 3:
        return "Rotten Apple"
    elif predictedClassIndex == 4:
        return "Rotten Banana"
    elif predictedClassIndex == 5:
        return "Rotten Orange"
    else:
        return "Unknown"

def test_image(test_image_path):
    model2 = YOLO('yolo-weights/yolov8n.pt')
    results = model2(test_image_path, stream=True)
    img = cv2.imread(test_image_path)
    pic = makePicture(test_image_path)
    for r in results:
        # put boxes to each fruit
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1
            canv = makeEmptyPicture(width, height)
            # copy the fruit to canvas and send it to predictedFruitAndState function
            for i in range(x1, x2):
                for j in range(y1, y2):
                    srcPixel = getPixel(pic, i, j)
                    srcCol = getColor(srcPixel)
                    targetPixel = getPixel(canv, i - x1, j - y1)
                    setColor(targetPixel, srcCol)
            writePictureTo(canv, r"C:/Users/i3mma/Documents/tempp.jpg")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            predictResult = predictedFruitAndState(r"C:/Users/i3mma/Documents/tempp.jpg")
            cvzone.putTextRect(img, f'{predictResult}',
                               (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
    cv2.imshow("test", img)
    cv2.waitKey(0)

test_image("C:/Users/i3mma/Downloads/testFruitImage.png")

#To save the Model
#model.save(os.path.join('models', 'fruitsRottenOrNot.h5'))

#To load the model
#model = load_model(os.path.join('models', 'fruitsRottenOrNot.h5'))
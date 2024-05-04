import os.path
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import cv2
import cvzone
from ultralytics import YOLO
from jes4py import *
image_path = "Testing_images/Rapple.jpg"
#Load the model
new_model = load_model(os.path.join('models', 'fruitsRottenOrNot.h5'))
def predictedFruitAndState(image_path):
    test_image = image.load_img(image_path, target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    test_image/=255
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
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
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            canv = makeEmptyPicture(x2 - x1, y2 - y1)
            for i in range(x1, x2):
                for j in range(y1, y2):
                    col = getColor(getPixel(pic, i, j))
                    setColor(getPixel(canv, i - x1, j - y1), col)
            writePictureTo(canv, r"C:/Users/i3mma/Documents/tempp.jpg")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            predictionResult = predictedFruitAndState(r"C:/Users/i3mma/Documents/tempp.jpg")
            cvzone.putTextRect(img, f'{predictionResult}',
                               (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
    cv2.imshow("test", img)
    cv2.waitKey(0)

print(test_image("Testing_images/Rbanana.jpg"))

from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


test_images = []
with open('input/test.csv', 'rt') as file:
    data = csv.reader(file)
    for row in data:
        img = row[0]
        test_images.append(img)

nrows = 150
ncolumns = 150
channels = 3

def read_process_image(image_list):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = [] # resized images
    y = [] # labels
    
    for image in image_list:
        exc = ""
        image_path = "input/train/images/" + image
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            X.append(cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            exc = image
            print(str(e))
    
    return X, y

X_test, y_test = read_process_image(test_images[0:10])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

model = load_model('model_keras.h5')

i = 0
columns = 5
text_labels = []
convert_label_dict = {'1': 'Cargo', 
                      '2': 'Military', 
                      '3': 'Carrier', 
                      '4': 'Cruise', 
                      '5': 'Tankers'}
plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    pred_label = np.argmax(pred)

    text_labels.append(str(pred_label))
    print(text_labels)
    plt.subplot(15 / columns + 1, columns, i + 1)
    plt.title('This is a ' + convert_label_dict.get(text_labels[i]))
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 20 == 0:
        break
plt.show()


from keras.models import load_model
import numpy as np
import cv2

model_path = "xception_Model_CLAHE.hdf5"
model = load_model(model_path)
print(model)
def model_predict(img_path, model):
    xtest_image = cv2.imread(img_path)
    xtest_image = cv2.resize(xtest_image, dsize=(100, 100))
    image = cv2.cvtColor(xtest_image,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    upd_img = cv2.merge((clahe_img,a,b))
    clahe_img_new = cv2.cvtColor(upd_img,cv2.COLOR_LAB2BGR)
    clahe_img_new = clahe_img_new.reshape(1, 100, 100, 3)
    preds = model.predict(clahe_img_new)
    preds = preds.flatten()
    if np.argmax(preds) == 0:
        prediction = "Covid"
    elif np.argmax(preds) == 1:
        prediction = "Non_Covid"
    else:
        prediction = "Normal"
    return prediction,preds

prediction,arr = model_predict("person1000_bacteria_2931.jpeg",model)
print(prediction)
print(arr)
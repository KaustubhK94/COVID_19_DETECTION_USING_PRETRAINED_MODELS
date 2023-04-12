from keras.models import load_model
import numpy as np
# from keras.preprocessing import image
# import tensorflow as tf
from keras.utils import load_img,img_to_array

model_path = "ResNet50_Model.hdf5"
model = load_model(model_path)
print(model)
def model_predict(img_path, model):
    xtest_image = load_img(img_path,target_size=(65,65))
    xtest_image = img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis=0)
    # xtest_image = xtest_image.reshape(224,224,3)
    preds = model.predict(xtest_image)
    preds = preds.flatten()
    if np.argmax(preds) == 0:
        prediction = "Covid"
    elif np.argmax(preds) == 1:
        prediction = "Non_Covid"
    else:
        prediction = "Normal"
    return prediction,preds

prediction,arr = model_predict("sub-S09669_ses-E21584_run-1_bp-chest_vp-ap_dx.png",model)
# prediction,arr = model_predict("person1000_bacteria_2931.jpeg",model)
print(prediction)
print(arr)
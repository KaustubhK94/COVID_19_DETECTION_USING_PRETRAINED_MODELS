import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
# from keras.utils import load_img, img_to_array
import numpy as np
import cv2

model_path = "xception_Model_CLAHE.hdf5"
model = load_model(model_path)

def model_predict(img_path, model):
    # xtest_image = load_img(img_path, target_size=(100,100,3))
    # xtest_image = img_to_array(xtest_image)
    xtest_image = cv2.imread(img_path)
    xtest_image = cv2.resize(xtest_image, dsize=(100, 100))
    # xtest_image = np.expand_dims(xtest_image, axis=0)
    image = cv2.cvtColor(xtest_image,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    upd_img = cv2.merge((clahe_img,a,b))
    clahe_img_new = cv2.cvtColor(upd_img,cv2.COLOR_LAB2BGR)
    clahe_img_new = clahe_img_new.reshape(1, 100, 100, 3)
    preds = model.predict(clahe_img_new)
    return preds
app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        preds = model_predict(img, model)
        if np.argmax(preds) == 0:
            prediction = "Covid"
        elif np.argmax(preds) == 1:
            prediction = "Result: Non-Covid"
        else:
            prediction = "Result: Normal"
        print(prediction)
        return render_template('index.html', img=img,  prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8001)

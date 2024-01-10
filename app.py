from flask import Flask, request, jsonify, render_template, redirect, flash
import secrets
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Load the pre-trained model
model = load_model('model.h5')
modelXray = load_model('xray_classifier.h5')


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET' :

        print("Inside GET method")
        return render_template("predict.html")


    if request.method == 'POST' :
        if 'xray' not in request.files:
            flash("Please upload an X-ray image")
            print("True")
            return render_template('predict.html')
            # return render_template('predict.html', 
            #                        negative_result='Please upload an X-ray image.')
        categories = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        file = request.files['xray']
        img = Image.open(file.stream)
        image = img.resize((224, 224))
        image = image.convert('RGB')
        # Load and preprocess the image
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image/255

        # Make predictions
        xray = modelXray.predict(image)
        print(xray)
        if (xray[0] > 0.5):
            print('This is a X-ray image')
            preds = model.predict(image)

            # Print the predicted category and corresponding probability
            for i in range(len(categories)):
                print(categories[i], ':', preds[0][i])

            # Get the predicted disease
            predicted_disease = categories[np.argmax(preds)]
            print('Predicted Disease:', predicted_disease)
            if (predicted_disease == "NORMAL"):
                # flash("Your lung us NORMAL!", "success")
                return render_template('predict.html',
                                    positive_result = 'This lung is NORMAL!')
            elif(predicted_disease == "COVID19"):
                return render_template('predict.html', 
                                    negative_result = 'The person with the given x - ray have COVID19!')
            elif(predicted_disease == "PNEUMONIA"):
                return render_template('predict.html', 
                                    negative_result = 'The person with the given x - ray have PNEUMONIA!')
            elif(predicted_disease == "TUBERCULOSIS"):
                return render_template('predict.html', 
                                    negative_result = 'The person with the given x - ray have TUBERCULOSIS!')
        else:
            print('This is not a X-ray image')
            return render_template('predict.html', 
                                    negative_result = 'Upload only X - ray images!')
        return redirect(url_for('home'))



if __name__ == '__main__':
    app.run(debug=True)
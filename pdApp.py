from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

wave_model = tf.keras.models.load_model('saved_models/wave_saved_model/model1024')


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/fog")
def fog():
    return render_template("fog.html")

@app.route("/fogPred", methods=["post"])
def fogPred():
    time = request.form.get("time")
    a1 = request.form.get("a1")
    a2 = request.form.get("a2")
    a3 = request.form.get("a3")
    t1 = request.form.get("t1")
    t2 = request.form.get("t2")
    t3 = request.form.get("t3")
    tr1 = request.form.get("tr1")
    tr2 = request.form.get("tr2")
    tr3 = request.form.get("tr3")

    inputs = np.array([time, a1,a2,a3,t1,t2,t3,tr1,tr2,tr3])
    inputs = np.expand_dims(inputs, axis = 0)
    print(inputs)

    loaded_model = pickle.load(open('saved_models/fog_dtc', 'rb'))

    pred = "No FOG" if np.squeeze(loaded_model.predict(inputs)).tolist() == 0 else "FOG"
    return render_template("fogPred.html", data=pred)   

@app.route("/audio")
def audio():
    return render_template("audio.html")

@app.route("/audioPred", methods=["post"])
def audioPred():
    i1 = request.form.get("i1")
    i2 = request.form.get("i2")
    i3 = request.form.get("i3")
    i4 = request.form.get("i4")
    i5 = request.form.get("i5")
    i6 = request.form.get("i6")
    i7 = request.form.get("i7")
    i8 = request.form.get("i8")
    i9 = request.form.get("i9")
    i10 = request.form.get("i10")
    i11 = request.form.get("i11")
    i12 = request.form.get("i12")
    i13 = request.form.get("i13")
    i14 = request.form.get("i14")
    i15 = request.form.get("i15")
    i16 = request.form.get("i16")
    i17 = request.form.get("i17")
    i18 = request.form.get("i18")
    i19 = request.form.get("i19")
    i20 = request.form.get("i20")
    i21 = request.form.get("i21")
    i22 = request.form.get("i22")
    
    list = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22]
    print("++++++++list",list)


    scalar = pickle.load(open('saved_models/audio_scaler', 'rb'))
    loaded_model = pickle.load(open('saved_models/audio_knn', 'rb'))

    inputs = np.array(list)
    inputs = np.expand_dims(inputs, axis = 0)

    inputs = scalar.transform(inputs)
    loaded_model = pickle.load(open('saved_models/fog_dtc', 'rb'))

    pred = "No PD" if np.squeeze(loaded_model.predict(inputs)).tolist() == 0 else "PD"

    return render_template("audioPred.html", data=pred)


@app.route("/hand")
def hand():
    return render_template("hand.html")

@app.route("/handPred", methods=["post"])
def prediction():

    wave_img = request.files["wave_img"]
    wave_img.save("uploads/wave_img.jpg")
    wave_img.save("static/images/wave_img.jpg")
    

    wave_img = image.load_img("uploads/wave_img.jpg", color_mode='rgb',target_size=(512, 1024))
    wave_img = image.img_to_array(wave_img)
    wave_img = wave_img/255.0
    wave_img = np.expand_dims(wave_img, axis = 0)
    wave_pred = wave_model.predict(wave_img, batch_size=1)
    wave_pred = np.squeeze(wave_pred)
    wave_pred = round(wave_pred.tolist()*100, 2)

    return render_template("handPred.html", data=wave_pred)


if __name__ == "__main__":
	app.run(debug=True)



    #source venv/bin/activate
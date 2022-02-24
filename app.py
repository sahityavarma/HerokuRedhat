from flask import Flask, jsonify,  request, render_template
#from sklearn.externals
import joblib
import numpy as np

import flask
app = Flask(__name__)
model_load = joblib.load("./models/DecisionTree.pkl")
min_max_scalar= joblib.load("./models/MinMaxScalar.pkl")

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
   if (request.method == 'POST'):
       final_features = [x for x in request.form.values()]
       value=min_max_scalar.transform(np.array(final_features).reshape(1, -1))
       output = model_load.predict(np.array(value).reshape(1, -1))
       if output[0]:
           prediction = "Not a Potential Customer"
       else:
           prediction = "Potential Customer"
       return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)









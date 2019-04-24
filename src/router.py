from __future__ import print_function
from future.standard_library import install_aliases
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from data.prediction import PREDICTION_DATA_PATH
import json
import os
import src.face as face

IMAGE_PREDICT = "image_predict"

install_aliases()
app = Flask(__name__)
cors = CORS(app)

predictor = None

@app.route('v1/api/train', endpoint='train', methods=['POST'])
def train():
    face.Face.train()
    global predictor
    predictor = face.Face()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('v1/api/predict', endpoint='predict', methods=['POST'])
def predict():
    global predictor
    if predictor is None:
        predictor = face.Face()
    file_predict = request.files[IMAGE_PREDICT]
    file_predict_path = os.path.join(PREDICTION_DATA_PATH, secure_filename(file_predict.filename))
    predictor.predict(file_predict_path)
    response = {'code': 0, 'data': {'id': 1, 'confidence': 0.5}}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('v1/api/member', methods=['POST'])
def add_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

@app.route('v1/api/member', methods=['PUT'])
def add_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('v1/api/member', methods=['DELETE'])
def add_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


if __name__ == '__main__':
    global f
    predictor = face.Face()
    with open('config.json') as json_data_file:
        cfg = json.load(json_data_file)
    port = int(os.getenv('PORT', cfg["port"]))
    print("Starting app on port %d" % port)
    app.run(threaded=True, debug=False, port=port,host = '0.0.0.0')
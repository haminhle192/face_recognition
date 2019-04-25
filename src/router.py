from __future__ import print_function
from future.standard_library import install_aliases
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from data.requesting import REQUESTING_DATA_PATH
import json
import os
import src.face as face

IMAGE_PREDICT = "image_predict"

install_aliases()
app = Flask(__name__)
cors = CORS(app)

predictor = None

@app.route('/v1/api/train', endpoint='train', methods=['POST'])
def train():
    face.Face.train()
    response = {'code': 0, 'message': 'Trained'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('/v1/api/predict', endpoint='predict', methods=['POST'])
def predict():

    file_predict = request.files[IMAGE_PREDICT]
    file_predict_path = os.path.join(REQUESTING_DATA_PATH, secure_filename(file_predict.filename))
    file_predict.save(file_predict_path)

    global predictor
    print('===============================')
    print(predictor)
    print('===============================')
    arr_obj = predictor.predict(file_predict_path)
    data = []
    for id_obj in arr_obj:
        user = {'id': id_obj, 'confidence': 1.0}
        data.append(user)

    response = {'code': 0, 'data': data}
    remove_file(file_predict_path)
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('/v1/api/member', methods=['POST'])
def add_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

@app.route('/v1/api/member', methods=['PUT'])
def update_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


@app.route('/v1/api/member', methods=['DELETE'])
def delete_member():
    face.Face.export_detection_for_training_data()
    response = {'code': 0, 'message': 'Entity Updated!'}
    res = json.dumps(response)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def remove_file(file_path):
    print("Deleting " + file_path + " Removed!")
    os.remove(file_path)
    print(file_path + " Removed!")

if __name__ == '__main__':
    global f
    predictor = face.Face()
    # predictor.predict(os.path.dirname(__file__) + "/../data/prediction/" + str(1) + ".png")
    print('===============================')
    print(predictor)
    print('===============================')
    with open('../config.json') as json_data_file:
        cfg = json.load(json_data_file)
    port = int(os.getenv('PORT', cfg["port"]))
    print("Starting app on port %d" % port)
    app.run(threaded=True, debug=False, port=port,host = '0.0.0.0')
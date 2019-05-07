from __future__ import print_function
from future.standard_library import install_aliases
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from data.prediction import PREDICTION_DATA_PATH
from data import DATA_PATH
from data.training_aligned import DATA_ALIGNED_PATH
import json
import os
import src.face as face
import time
import shutil

IMAGE_PREDICT = "image_predict"
IMAGE_MEMBER = "image_training"
FACE_ID = "face_id"

install_aliases()
app = Flask(__name__)
cors = CORS(app)

training = False
predictor = None
current_milli_time = lambda: int(round(time.time() * 1000))


@app.route('/v1/api/train', endpoint='train', methods=['POST'])
def train():
    global training
    if training is True:
        response = generate_response(1006, "Bot is training")
        return response
    training = True
    start_time = time.time()
    face.Face.train(data_dir=DATA_ALIGNED_PATH)
    global predictor
    predictor = None
    predictor = face.Face()
    duration = "Time to train: %s" % str(time.time() - start_time)
    training = False
    response = generate_response(0, duration)
    return response


@app.route('/v1/api/predict', methods=['POST'])
def predict():
    file_predict = request.files[IMAGE_PREDICT]
    mime_type = file_predict.content_type
    if mime_type not in ['image/png', 'image/jpg']:
        response = generate_response(1001, 'Image have to png or jpg format')
        return response

    file_name = generate_name(secure_filename(file_predict.filename))
    file_predict_path = os.path.join(PREDICTION_DATA_PATH, file_name)
    file_predict.save(file_predict_path)
    start_time = time.time()
    global predictor
    arr_obj = predictor.predict(file_predict_path)
    data = []
    for id_obj in arr_obj:
        user = {'id': id_obj}
        data.append(user)
    duration = "Predict: %s" % str(time.time() - start_time)
    os.remove(file_predict_path)
    response = generate_response(0, duration, data)
    return response


@app.route('/v1/api/member', methods=['POST'])
def add_member():
    face_id = request.form[FACE_ID]
    existed_member_path = os.path.join(DATA_ALIGNED_PATH, face_id)
    if os.path.exists(existed_member_path):
        response = generate_response(1002, 'Member is existed. Could not create new member')
        return response

    result = align_face(face_id)
    if result is None:
        response = generate_response(1005, 'Not found image. Plz check your request!')
        return response

    source_dir = os.path.join(result.get("output_path"), face_id)
    shutil.move(source_dir, DATA_ALIGNED_PATH)
    shutil.rmtree(result.get("input_path"))
    shutil.rmtree(result.get("output_path"))
    response = generate_response(0, 'Face uploaded')
    return response


@app.route('/v1/api/member', methods=['PUT'])
def update_member():
    face_id = request.form[FACE_ID]
    existed_member_path = os.path.join(DATA_ALIGNED_PATH, face_id)
    if not os.path.exists(existed_member_path):
        response = generate_response(1003, 'Member is not existed. Could not update member')
        return response

    result = align_face(face_id)
    if result is None:
        response = generate_response(1005, 'Not found image. Plz check your request!')
        return response

    source_aligned_path = os.path.join(result.get("output_path"), face_id)
    destination_aligned_path = os.path.join(DATA_ALIGNED_PATH, face_id)
    move_file(source_aligned_path, destination_aligned_path)
    shutil.rmtree(result.get("input_path"))
    shutil.rmtree(result.get("output_path"))
    response = generate_response(0, 'Face updated!')
    return response


@app.route('/v1/api/member', methods=['DELETE'])
def delete_member():
    face_id = request.form[FACE_ID]
    existed_member_path = os.path.join(DATA_ALIGNED_PATH, face_id)
    if not os.path.exists(existed_member_path):
        response = generate_response(1004, 'Member is not existed. Could not delete member')
        return response
    destination_path = os.path.join(DATA_ALIGNED_PATH, face_id)
    shutil.rmtree(destination_path)
    response = generate_response(0, 'Deleted')
    return response


def align_face(face_id):
    folder_input_name = generate_name("{}_{}".format(face_id, "input"))
    folder_output_name = generate_name("{}_{}".format(face_id, "output"))

    tmp_input_face_path = os.path.join(DATA_PATH, folder_input_name , face_id)
    os.makedirs(tmp_input_face_path)

    tmp_output_path = os.path.join(DATA_PATH, folder_output_name)
    os.makedirs(tmp_output_path)
    found_file = False
    for key in request.files.keys():
        file_request = request.files[key]
        mime_type = file_request.content_type
        if mime_type not in ['image/png', 'image/jpg']:
            print(mime_type)
            continue
        found_file = True
        file_name = generate_name(secure_filename(file_request.filename))
        file_path = os.path.join(tmp_input_face_path, file_name)
        file_request.save(file_path)

    if not found_file:
        return None

    tmp_input_path = os.path.join(DATA_PATH, folder_input_name)
    face.Face.export_detection_for_training_data(input_dir=tmp_input_path, output_dir=tmp_output_path)
    return {"input_path": tmp_input_path, "output_path": tmp_output_path}

def generate_response(code=0, message='', data=None):
    response = {'code': code, 'message': message, 'data': data}
    res = json.dumps(response)
    response = make_response(res)
    response.headers['Content-Type'] = 'application/json'
    return response


def generate_name(suffix):
    timestamp = current_milli_time()
    return "{}_{}".format(timestamp, suffix)


def remove_file(file_path):
    print("Deleting " + file_path + " Removed!")
    os.remove(file_path)
    print(file_path + " Removed!")


def move_file(source_folder, destination_folder):
    files = os.listdir(source_folder)
    for _file in files:
        file_path = "{}/{}".format(source_folder, _file)
        shutil.move(file_path, destination_folder)

if __name__ == '__main__':
    start_time = time.time()
    global f
    predictor = face.Face()
    print("Time to load model: %s" % str(time.time() - start_time))
    # path = ROOT_PATH
    with open('config.json') as json_data_file:
        cfg = json.load(json_data_file)
    port = int(os.getenv('PORT', cfg["port"]))
    print("Starting app on port %d" % port)
    app.run(threaded=True, debug=False, port=port,host = '0.0.0.0')
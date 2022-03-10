from __future__ import annotations
from ast import Global
import json
from turtle import left

import cv2
import mediapipe as mp
# import transformations as tf

from iottalkpy.dan import NoData
from google.protobuf.json_format import MessageToDict

import os, time, csv, re
import numpy as np
from queue import Queue
import math as m
import threading

import itertools
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class CameraOpenError(Exception):
    pass

### The registeration api url, you can use IP or Domain.
api_url = 'http://140.113.110.16:10001/csm'  # default
# api_url = 'http://localhost/csm'  # with URL prefix
# api_url = 'http://localhost:9992/csm'  # with URL prefix + port

### [OPTIONAL] If not given or None, server will auto-generate.
device_name = 'mp'

### [OPTIONAL] If not given or None, DAN will register using a random UUID.
### Or you can use following code to use MAC address for device_addr.
# from uuid import getnode
# device_addr = "{:012X}".format(getnode())
# device_addr = "453013fb-5933-4a86-bccc-33efcdce5fc2"

### [OPTIONAL] If the device_addr is set as a fixed value, user can enable
### this option and make the DA register/deregister without rebinding on GUI
# persistent_binding = True

### [OPTIONAL] If not given or None, this device will be used by anyone.
# username = 'myname'

### The Device Model in IoTtalk, please check IoTtalk document.
device_model = 'MPdevice'
# device_model = 'MPdevice'

### The input/output device features, please check IoTtalk document.
idf_list = [ 
            # Face
            'FLeftCheek-I',
            'FLeftEyebrow-I',
            'FLeftEyeLower-I',
            'FLeftEyeUpper-I',
            'FLipsLower-I',
            'FLipsUpper-I',
            'FNose-I',
            'FRightCheek-I',
            'FRightEyebrow-I',
            'FRightEyeLower-I',
            'FRightEyeUpper-I',
            'FSilhouette-I',
            # Hand Coordinate
            'HLeftIndexCoordinate-I',
            'HLeftMiddleCoordinate-I',
            'HLeftPinkyCoordinate-I',
            'HLeftRingCoordinate-I',
            'HLeftThumbCoordinate-I',
            'HLeftWristCoordinate-I',
            'HRightIndexCoordinate-I',
            'HRightMiddleCoordinate-I',
            'HRightPinkyCoordinate-I',
            'HRightRingCoordinate-I',
            'HRightThumbCoordinate-I',
            'HRightWristCoordinate-I',
            # Hnad Angle
            'HLeftIndex-I',
            'HLeftIndexAngle-I',
            'HLeftMiddleAngle-I',
            'HLeftPinkyAngle-I',
            'HLeftRingAngle-I',
            'HLeftThumbAngle-I',
            'HLeftWristAngle-I',
            'HRightIndexAngle-I',
            'HRightMiddleAngle-I',
            'HRightPinkyAngle-I',
            'HRightRingAngle-I',
            'HRightThumbAngle-I',
            'HRightWristAngle-I',
            # Pose
            'PHead-I',
            'PLeftLowerLimb-I',
            'PLeftUpperLimb-I',
            'PRightLowerLimb-I',
            'PRightUpperLimb-I',
            ]

### Set the push interval, default = 1 (sec)
### Or you can set to 0, and control in your feature input function.
push_interval = 1  # global interval

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

face_visibility = False
left_hand_visibility = False
right_hand_visibility = False

face_dict = {}
left_hand_dict = {}
right_hand_dict = {}
pose_dict = {}

# hand
wrist_idx = [0]
thumb_idx = [0, 2, 3, 4]
index_idx =[0, 5, 6, 7, 8]
middle_idx = [0, 9, 10, 11, 12]
ring_idx = [0, 13, 14, 15, 16]
pinky_idx = [0, 17, 18, 19, 20]
# Pose
head_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lu_limb_idx = [11, 13, 15, 17, 19, 21]
ru_limb_idx = [12, 14, 16, 18, 20, 22]
ll_limb_idx = [23, 25, 27, 29, 31]
rl_limb_idx = [24, 26, 28, 30, 32]
# Face
l_cheek_idx = [425]
l_eyebrow_idx = [276, 283, 282, 295, 285]
l_eye_lower_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362]
l_eye_upper_idx = [466, 388, 387, 386, 385, 384, 398]
lip_lower_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lip_upper_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
# midwayBetweenEyes: [168] noseTip: [1] noseBottom:[2] noseRightCorner: [98] noseLeftCorner: [327]
nose_idx = [1, 2, 98, 168, 327]
r_cheek_idx = [205]
r_eyebrow_idx = [ 46, 53,  52,   65,  55]
r_eye_lower_idx = [ 33, 7, 163, 144, 145, 153, 154, 155, 133]
r_eye_upper_idx = [ 246, 161, 160, 159, 158, 157, 173]
silhouette_idx = [   10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172,  58, 132,  93, 234, 127, 162, 21,  54,  103, 67,  109]

record_flag = False
train_flag = True
image_id = 0
annotation = None
keypoint_classifier = None
gesture = None
hand_sign_id = [-1, -1]

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='./keypoint_classifier_new.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        np.set_printoptions(precision=4, suppress=True)
        result_index = np.argmax(np.squeeze(result))
        #print(np.squeeze(result))
        if np.squeeze(result)[result_index] < 0.5:
            result_index = -1
        return result_index

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break
    # retrieve latest frame
    def read(self):
        ret, frame = self.cap.retrieve()
        return ret, frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = []

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point.x, landmark_point.y

        temp_landmark_list.append([landmark_point.x - base_x, landmark_point.y - base_y])

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def singleLMtoDegreeUsingLawOfCosines(pa, pb, pc):

    ab_y = float(pb[1])-float(pa[1])
    ab_x = float(pb[0])-float(pa[0])
    ab_z = float(pb[2])-float(pa[2])
    bc_x = float(pc[0])-float(pb[0])
    bc_y = float(pc[1])-float(pb[1])
    bc_z = float(pc[2])-float(pb[2])
    ac_x = float(pc[0])-float(pa[0])
    ac_y = float(pc[1])-float(pa[1])
    ac_z = float(pc[2])-float(pa[2])
    a = m.sqrt(m.pow(ab_x, 2)+m.pow(ab_y, 2)+m.pow(ab_z, 2))    # AB = a
    b = m.sqrt(m.pow(bc_x, 2)+m.pow(bc_y, 2)+m.pow(bc_z, 2))    # BC = b
    c = m.sqrt(m.pow(ac_x, 2)+m.pow(ac_y, 2)+m.pow(ac_z, 2))    # AC = c
    cos = (m.pow(a, 2)+m.pow(b, 2)-m.pow(c, 2))/(2*a*b)         # cos = (a^2 + b^2 - c^2) / 2ab
    degree = m.degrees(m.acos(cos))                             # degree = arcccos( cos )

    return degree

def LMtoRotation(landmarks_mediapipe):

    lm = landmarks_mediapipe

    pitch_pa = [lm[0]['x'], lm[0]['y'], lm[5]['z']]
    pitch_pb = [lm[0]['x'], lm[0]['y'], lm[0]['z']]
    pitch_pc = [lm[5]['x'], lm[5]['y'], lm[5]['z']]

    roll_pa = [         0, lm[0]['y'], lm[0]['z']]
    roll_pb = [lm[0]['x'], lm[0]['y'], lm[0]['z']]
    roll_pc = [lm[5]['x'], lm[5]['y'], lm[0]['z']]

    yaw_pa = [ lm[5]['x'], lm[5]['y'], lm[17]['z']]
    yaw_pb = [lm[17]['x'], lm[5]['y'], lm[17]['z']]
    yaw_pc = [ lm[5]['x'], lm[5]['y'],  lm[5]['z']]

    pitch = singleLMtoDegreeUsingLawOfCosines(pitch_pa, pitch_pb, pitch_pc)
    pitch = 90 - pitch
    if lm[8]['z'] < lm[0]['z']:
        pitch = -pitch

    yaw = singleLMtoDegreeUsingLawOfCosines(yaw_pa, yaw_pb, yaw_pc)
    if lm[5]['z'] > lm[17]['z']:
        yaw = -yaw

    roll = singleLMtoDegreeUsingLawOfCosines(roll_pa, roll_pb, roll_pc)

    return [pitch, yaw, roll]

def set_record(cmd, device_name, uuid):
    global record_flag, image_id, train_flag, keypoint_classifier, gesture
    print(cmd)
    if 'unrecord' == cmd:
        record_flag = False
    elif 'record' == cmd:
        path = os.path.join(os.getcwd(), 'media', '{}-{}'.format(device_name, uuid))
        if not os.path.exists(path):
            os.mkdir(path)
            annotation = open(os.path.join( path, 'annotation.csv'), 'w', newline='')
            annotation.close()
        image_id = len(os.listdir(path))
        record_flag = True
    elif 'train':
        train_flag = True
    elif 'endTrain':
    # get keypoint tflite model 
        tflite = os.path.join(os.getcwd(), 'media', "{}-{}".format(device_name, uuid),'keypoint_classifier.tflite')
        dataset = os.path.join(os.getcwd(), 'media', "{}-{}".format(device_name, uuid),'keypoint.csv')

        if os.path.isfile(tflite):
            keypoint_classifier = KeyPointClassifier(model_path=tflite)
            y_dataset = np.loadtxt(dataset, delimiter=',', dtype='str', usecols=(42))
            labelencoder = LabelEncoder()
            y_dataset = labelencoder.fit_transform(y_dataset)
            gesture = labelencoder.classes_
        train_flag = False
    else:
        pass

def save_imgae(image, device_name, uuid, hand_landmark):
    global image_id
    path = os.path.join(os.getcwd(), 'media')
    path = os.path.join(path, '{}-{}'.format(device_name, uuid))
    anno_path = os.path.join( path, 'annotation.csv')
    image_path = os.path.join(path, '{}.png'.format(str(image_id).zfill(5)))
    
    with open(anno_path, 'a+', newline='') as annotation:
        ann_writer = csv.writer(annotation)
        if hand_landmark[0]:
            temp_landmark_list=[]
            temp_landmark_list.append(image_id)
            for landmark in hand_landmark[2]['landmark']:
                temp_landmark_list.append(landmark['x'])
                temp_landmark_list.append(landmark['y'])
            temp_landmark_list.append("no_label")
            ann_writer.writerow(temp_landmark_list)

        if hand_landmark[1]:
            temp_landmark_list=[]
            temp_landmark_list.append(image_id)
            for landmark in hand_landmark[3]['landmark']:
                temp_landmark_list.append(landmark['x'])
                temp_landmark_list.append(landmark['y'])
            temp_landmark_list.append("no_label")
            ann_writer.writerow(temp_landmark_list)

    if hand_landmark[0] or hand_landmark[1]:
        cv2.imwrite(image_path, image)
        image_id+=1

# models = ['Hands', 'Pose', 'Face', 'Holistic']
def StreamingHands(url, device_name, model, model_complexity, detection_confidence, uuid):
    global right_hand_visibility, face_visibility, left_hand_visibility, record_flag, train_flag
    global face_dict, left_hand_dict, right_hand_dict, pose_dict, hand_sign_id, keypoint_classifier, gesture

    # Device Frame to Rtsp function, will update...
    # -------------------------------------------------------
    # out = cv2.VideoWriter('appsrc ! videoconvert' + \
    #     ' ! x264enc speed-preset=ultrafast bitrate=600' + \
    #     ' ! rtspclientsink location=rtsp://localhost:8554/{}'.format(device_name),
    #     cv2.CAP_GSTREAMER, 0, 30, (960, 540), True)
    # if not out.isOpened():
    #     raise Exception("can't open video writer")

    cap = VideoCapture(url)
    if not cap.isOpened():
        raise CameraOpenError("Camera Not Open")

    with mp_hands.Hands( 
        min_detection_confidence=detection_confidence, 
        min_tracking_confidence=0.5, 
        model_complexity=model_complexity) as hands:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success:
                # print("Ignoring empty camera frame.")
                continue
            
            hand_landmark = []

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            left_hand_visibility = False
            right_hand_visibility = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for idx, handLms in enumerate(results.multi_hand_landmarks):
                    lbl = results.multi_handedness[idx].classification[0].label
                    if lbl == 'Left':
                        ls = []
                        for l in handLms.landmark:
                            ls.append({'x':l.x, 'y':l.y, 'z':l.z})
                        left_hand_dict['landmark'] = ls
                        left_hand_visibility = True
                    else:
                        ls = []
                        for l in handLms.landmark:
                            ls.append({'x':l.x, 'y':l.y, 'z':l.z})
                        right_hand_dict['landmark'] = ls
                        right_hand_visibility = True
            # image = cv2.flip(image, 1)

            if record_flag:
                save_imgae(image, device_name, uuid, (right_hand_visibility, left_hand_visibility, right_hand_dict, left_hand_dict))
            if not train_flag:
                hand_sign_id = [-1, -1]
                if results.multi_hand_world_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_world_landmarks):
                        if results.multi_handedness[idx].classification[0].label == 'Right':
                            lbl = 0
                        else:
                            lbl = 1
                        pre_processed_landmark_list = pre_process_landmark(hand_landmarks.landmark)
                        hand_sign_id[lbl] = int(keypoint_classifier(pre_processed_landmark_list))

            if hand_sign_id[0] != -1:
                cv2.putText(image, f'gesture : {gesture[hand_sign_id[0]]}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            # cv2.imshow('MediaPipe Hands', image)
            image = cv2.resize(image, (960, 540), interpolation = cv2.INTER_LINEAR)
            
            # Device Frame to Rtsp function, will update...
            # ---------------------------------------------
            #out.write(image)
            
            # out.write(image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def Streaming(url, device_name, model, model_complexity, detection_confidence, uuid):
    global right_hand_visibility, face_visibility, left_hand_visibility, record_flag
    global face_dict, left_hand_dict, right_hand_dict, pose_dict

    cap = VideoCapture(url)
    if not cap.isOpened():
        raise CameraOpenError("Camera Not Open")

    with mp_holistic.Holistic( 
        min_detection_confidence=detection_confidence, 
        min_tracking_confidence=0.5, 
        model_complexity=model_complexity) as holistic:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success:
                # print("Ignoring empty camera frame.")
                continue
            
            hand_landmark = []

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                face_dict = MessageToDict(results.face_landmarks)
                face_visibility = True
            else:
                face_visibility = False
            
            if results.left_hand_landmarks:
                left_hand_dict = MessageToDict(results.left_hand_landmarks)
                left_hand_visibility = True
            else:
                left_hand_visibility = False

            if results.right_hand_landmarks:
                right_hand_dict = MessageToDict(results.right_hand_landmarks)
                right_hand_visibility = True
            else:
                right_hand_visibility = False
                
            if results.pose_landmarks:
                pose_dict = MessageToDict(results.pose_landmarks)

            if (model == 'Face' or model == 'Holistic'):
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list= results.face_landmarks,
                #     connections=mp_holistic.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())
            if (model == 'Pose' or model == 'Holistic'):
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
            
            if results.right_hand_landmarks and (model == 'Hands' or model == 'Holistic'):
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if results.left_hand_landmarks and (model == 'Hands' or model == 'Holistic'):
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            image = cv2.flip(image, 1)
            # cv2.imshow('MediaPipe Holistic', image)
            if record_flag:
                save_imgae(image, device_name, uuid, (right_hand_visibility, left_hand_visibility, right_hand_dict, left_hand_dict))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def on_register(dan):
    print('register successfully')

def to_vector(coordinate):
    return np.array([coordinate['x'], coordinate['y'], coordinate['z']])

def coordinate_to_angle(coordinate):
    angles = []
    for i in range(len(coordinate)-2):
        v1 = to_vector(coordinate[i]) - to_vector(coordinate[i+1])
        v2 = to_vector(coordinate[i+2]) - to_vector(coordinate[i+1])

        Lv1=np.sqrt(v1.dot(v1))
        Lv2=np.sqrt(v2.dot(v2))
        cos_angle=v1.dot(v2)/(Lv1*Lv2)
        angle_rad=np.arccos(cos_angle)
        angle=angle_rad*360/2/np.pi

        angles.append(angle)
    
    return angles
# -------------------------------------------------------------------------------
#                           Face IDF Function
# -------------------------------------------------------------------------------
def FLeftCheek_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in l_cheek_idx])                
    else:
        return NoData()

def FLeftEyebrow_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in l_eyebrow_idx])
    else:
        return NoData()

def FLeftEyeLower_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in l_eye_lower_idx])
    else:
        return NoData()

def FLeftEyeUpper_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in l_eye_upper_idx])
    else:
        return NoData()

def FLipsLower_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in lip_lower_idx])
    else:
        return NoData()

def FLipsUpper_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in lip_upper_idx])
    else:
        return NoData()

def FNose_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in nose_idx])
    else:
        return NoData()

def FRightCheek_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in r_cheek_idx])
    else:
        return NoData()

def FRightEyebrow_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in r_eyebrow_idx])
    else:
        return NoData()

def FRightEyeLower_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in r_eye_lower_idx])
    else:
        return NoData()

def FRightEyeUpper_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in r_eye_upper_idx])
    else:
        return NoData()

def FSilhouette_I():
    if face_visibility:
        return json.dumps([face_dict['landmark'][x] for x in silhouette_idx])
    else:
        return NoData()
# -------------------------------------------------------------------------------
#                           Hand Angle IDF Function
# -------------------------------------------------------------------------------
def HLeftIndex_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HLeftIndexAngle_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HLeftMiddleAngle_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HLeftPinkyAngle_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HLeftRingAngle_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HLeftThumbAngle_I():
    if left_hand_visibility:
        return coordinate_to_angle([left_hand_dict['landmark'][x] for x in ([0,1] + thumb_idx[-3:])])
    else:
        return NoData()

def HLeftWristAngle_I():
    if left_hand_visibility:
        return LMtoRotation(left_hand_dict['landmark'])
    else:
        return NoData()

def HRightIndexAngle_I():
    if right_hand_visibility:
        return coordinate_to_angle([right_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HRightMiddleAngle_I():
    if right_hand_visibility:
        return coordinate_to_angle([right_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HRightPinkyAngle_I():
    if right_hand_visibility:
        return coordinate_to_angle([right_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HRightRingAngle_I():
    if right_hand_visibility:
        return coordinate_to_angle([right_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HRightThumbAngle_I():
    if right_hand_visibility:
        return coordinate_to_angle([right_hand_dict['landmark'][x] for x in ([0,1] + thumb_idx[-3:])])
    else:
        return NoData()

def HRightWristAngle_I():
    if right_hand_visibility:
        return LMtoRotation(right_hand_dict['landmark'])
    else:
        return NoData()
# -------------------------------------------------------------------------------
#                           Hand Coordinate IDF Function
# -------------------------------------------------------------------------------
def HLeftIndexCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HLeftMiddleCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HLeftPinkyCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HLeftRingCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HLeftThumbCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in thumb_idx])
    else:
        return NoData()

def HLeftWristCoordinate_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in wrist_idx])
    else:
        return NoData()

def HRightIndexCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HRightMiddleCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HRightPinkyCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HRightRingCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HRightThumbCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in thumb_idx])
    else:
        return NoData()

def HRightWristCoordinate_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in wrist_idx])
    else:
        return NoData()
# -------------------------------------------------------------------------------
#                           Pose IDF Function
# -------------------------------------------------------------------------------
def PHead_I():
    return json.dumps([pose_dict['landmark'][x] for x in head_idx])

def PLeftLowerLimb_I():
    return json.dumps([pose_dict['landmark'][x] for x in ll_limb_idx])

def PLeftUpperLimb_I():
    return json.dumps([pose_dict['landmark'][x] for x in lu_limb_idx])

def PRightLowerLimb_I():
    return json.dumps([pose_dict['landmark'][x] for x in rl_limb_idx])

def PRightUpperLimb_I():
    return json.dumps([pose_dict['landmark'][x] for x in ru_limb_idx])


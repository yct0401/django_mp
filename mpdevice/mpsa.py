import json

import cv2
import mediapipe as mp

from iottalkpy.dan import NoData
from google.protobuf.json_format import MessageToDict

import os, time
import tempfile

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
# device_addr = "..."

### [OPTIONAL] If the device_addr is set as a fixed value, user can enable
### this option and make the DA register/deregister without rebinding on GUI
# persistent_binding = True

### [OPTIONAL] If not given or None, this device will be used by anyone.
# username = 'myname'

### The Device Model in IoTtalk, please check IoTtalk document.
device_model = 'MediapipeDevice'

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
            # Hand
            'HLeftIndex-I',
            'HLeftMiddle-I',
            'HLeftPinky-I',
            'HLeftRing-I',
            'HLeftThumb-I',
            'HLeftWrist-I',
            'HRightIndex-I',
            'HRightMiddle-I',
            'HRightPinky-I',
            'HRightRing-I',
            'HRightThumb-I',
            'HRightWrist-I',
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

face_visibility = False
left_hand_visibility = False
right_hand_visibility = False
save_imgae = True

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

holistic = mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

images = []
fp = tempfile.NamedTemporaryFile('w+t')

def save_imgae( path="./", name="img", id=0):
    global images
    orign_path = os.path.join(path, (name + "_" + str(id) + "_orign.png"))
    predetion_path = os.path.join(path, (name + "_" + str(id) + "_predetion.png"))
    print(len(images))
    if len(images) == 2:
        cv2.imwrite(orign_path, images[0])
        cv2.imwrite(predetion_path, images[1])
        return True
    return False

def Streaming(url, device_name):
    global right_hand_visibility, face_visibility, left_hand_visibility
    global face_dict, left_hand_dict, right_hand_dict, pose_dict, images

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise CameraOpenError("Camera Not Open")
    while cap.isOpened():
        
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if len(images) == 2:
            images[0] = cv2.flip(frame, 1)
        else :
            images.append(cv2.flip(frame, 1))

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                
        image.flags.writeable = False
        results = holistic.process(image)

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
            

        # print('right_hand_visibility out = ', right_hand_visibility)
        if results.pose_landmarks:
            pose_dict = MessageToDict(results.pose_landmarks)

        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style())

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style())
        # cv2.imshow('MediaPipe Holistic', image)
        # time.sleep(0.1)
        # if len(images) == 2:
        #     images[1] = image
        # else :
        #     images.append(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    fp.close()
    cap.release()

def on_register(dan):
    print('register successfully')


# Face IDF Function
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

# Hand IDF Function
def HLeftIndex_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HLeftMiddle_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HLeftPinky_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HLeftRing_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HLeftThumb_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in thumb_idx])
    else:
        return NoData()

def HLeftWrist_I():
    if left_hand_visibility:
        return json.dumps([left_hand_dict['landmark'][x] for x in wrist_idx])
    else:
        return NoData()

def HRightIndex_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in index_idx])
    else:
        return NoData()

def HRightMiddle_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in middle_idx])
    else:
        return NoData()

def HRightPinky_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in pinky_idx])
    else:
        return NoData()

def HRightRing_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in ring_idx])
    else:
        return NoData()

def HRightThumb_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in thumb_idx])
    else:
        return NoData()

def HRightWrist_I():
    if right_hand_visibility:
        return json.dumps([right_hand_dict['landmark'][x] for x in wrist_idx])
    else:
        return NoData()

# Pose IDF Function
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


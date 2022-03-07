from email.policy import default
from tkinter.messagebox import NO
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from django.templatetags.static import static
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from matplotlib.style import context

import numpy as np
import subprocess, os, signal, sys
import io
import base64

from queue import Queue, Empty
from threading import Thread, current_thread

from uuid import uuid4
from rtsp.models import RTSP, device, project
from mpdevice import settings
import pandas as pd
import itertools

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

train_flag = False
plt.switch_backend('SVG')

models = ['Hands', 'Pose', 'Face', 'Holistic']

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
        # print(np.squeeze(result))
        if np.squeeze(result)[result_index] < 0.5:
            result_index = -1
        return result_index

def pre_process_landmark(landmark_list):
    temp_landmark_list = []

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point.x, landmark_point.y

        temp_landmark_list.append([landmark_point.x - base_x, landmark_point.y - base_y])

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

class MPdevice(object):
    def __init__(self):
        self.pipe_w=None
        self.pipe_r=None
        self.output_file=None
        self.process=None
        self.uuid=None
    
    def __del__(self):
        self.output_file.close()

class LiveWebCam(object):
    def __init__(self, url):
        self.url = url
        self.cam = cv2.VideoCapture(url)
        self.active = self.cam.isOpened()
    
    def __del__(self):
        if self.cam:
            self.cam.release()

    def set_cam(self, url):
        self.cam = cv2.VideoCapture(url)
        self.url = url
        self.active = self.cam.isOpened()

    def get_frame(self, model, models_function, keypoint_classifier=None, gesture=None):
        success, imgNp = self.cam.read()
        if not success:
            image = np.array([[[0]*640]*480]*3)
            ret, jpeg = cv2.imencode('.png', image)
            return jpeg

        if model=='Hands':
            image = cv2.cvtColor(cv2.flip(imgNp, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor( imgNp, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = models_function.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hand_sign_id = -1

        if model == 'Hands':
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                if keypoint_classifier and not train_flag:
                    for idx, hand_landmarks in enumerate(results.multi_hand_world_landmarks):
                        if results.multi_handedness[idx].classification[0].label == 'Right':
                            pre_processed_landmark_list = pre_process_landmark(hand_landmarks.landmark)
                            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        elif model == 'Pose':
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())            
        elif model == 'Face':
            if results.multi_face_landmarks:
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
        elif model == 'Holistic':
            if results.face_landmarks:
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
            

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
        else:
            pass

        if hand_sign_id != -1:
            cv2.putText(image, f'gesture : {gesture[hand_sign_id]}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        height = int(960 / image.shape[1] * image.shape[0])
        image = cv2.resize(image, (960, height), interpolation = cv2.INTER_LINEAR)
        
        ret, jpeg = cv2.imencode('.png', image)
        return jpeg.tobytes()

CAM = {}
id_pool = [0]
p = {}

device.objects.all().delete()
for rtsp in  RTSP.objects.all():
    CAM[rtsp.name] = LiveWebCam(rtsp.url)


# --------------------------------------------
#                   train
# --------------------------------------------

# collect stdout from thread
class Printer:
    def __init__(self):
        self.queues = {}

    def write(self, value):
        '''handle stdout'''
        queue = self.queues.get(current_thread().name)
        if queue:
            queue.put(value)
        else:
            sys.__stdout__.write(value)

    def flush(self):
        '''Django would crash without this'''
        pass

    def register(self, thread):
        queue = Queue()
        self.queues[thread.name] = queue
        return queue

    def clean(self, thread):
        del self.queues[thread.name]

printer = Printer()
sys.stdout = printer

class Steamer:
    def __init__(self, target, args):
        self.thread = Thread(target=target, args=args)
        self.queue = printer.register(self.thread)

    def start(self, device_name):
        global p, train_flag
        self.thread.daemon = True
        self.thread.start()
        while self.thread.is_alive():
            try:
                item = self.queue.get()
                if item == 'end':
                    break
                yield f'{item}<br>'
            except Empty:
                pass
        main_page_url = reverse('main_page')
        yield '<a href=\"{}\" >\
                    <input type=\"button\" id=\"btn\" value=\"Done\" style=\"margin:50px;\">\
                </a>'.format(main_page_url)
        printer.clean(self.thread)
        
        train_flag = False

        d = device_name.split('-')
        if d[0] in p:
            if p[d[0]].uuid == d[1]:
                print('endTrain', file=p[d[0]].output_file)
            else:
                print("diff dev")
        else:
            print("dev not exist")

def train_model(device_name):
    RANDOM_SEED = 100
    
    dataset = os.path.join(settings.BASE_DIR, 'media', device_name,'keypoint.csv')
    model_save_path = os.path.join(settings.BASE_DIR, 'media', device_name,'keypoint_classifier.hdf5')
    tflite_save_path = os.path.join(settings.BASE_DIR, 'media', device_name,'keypoint_classifier.tflite')

    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(0, 21 * 2)))

    for row in range(len(X_dataset)):
        base_x = X_dataset[row][0]
        base_y = X_dataset[row][1]
        for idx in range(0, len(X_dataset[row]), 2):
            X_dataset[row][idx]  -=base_x
            X_dataset[row][idx+1]-=base_y
        max_value = max(list(map(abs, X_dataset[row])))
        def normalize_(n):
            return n / max_value
        X_dataset[row] = list(map(normalize_, X_dataset[row]))
    
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='str', usecols=(42))
    labelencoder = LabelEncoder()
    y_dataset = labelencoder.fit_transform(y_dataset)

    NUM_CLASSES = len(labelencoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    # Dropout randomly sets input units to 0 with a frequency of rate
    layer3 = NUM_CLASSES*4
    layer5 = NUM_CLASSES*2
    if NUM_CLASSES < 5:
        layer3 = 20
        layer5 = 10

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(layer3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(layer5, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.summary()

    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=0, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    class myuCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("End epoch {} of training - loss: {:7.4f} accuracy: {:7.4f} - val_loss: {:7.4f} - val_accuracy: {:7.4f} ".format(
                epoch, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']))

    c = myuCallback()

    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    history = model.fit(
            X_train,
            y_train,
            epochs=1000,
            batch_size=128,
            verbose=0,
            validation_data=(X_test, y_test),
            callbacks=[c, es_callback, cp_callback]
        )

    # Model evaluation
    print("Model evaluation")
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

    s = io.BytesIO()
    plt.subplots_adjust(wspace =0, hspace =0.5)
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2 = plt.subplot(2, 1, 2)

    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    plt.savefig(s, format='png', bbox_inches="tight")
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    print('<img src="data:image/png;base64,%s">' % s)



    def print_confusion_matrix(y_true, y_pred, report=True):

        s1 = io.BytesIO()

        labels = sorted(list(set(y_true)))
        cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
        
        df_cmx = pd.DataFrame(cmx_data, index=[ labelencoder.classes_[x] for x in labels], columns=[ labelencoder.classes_[x] for x in labels])
    
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
        ax.set_ylim(len(set(y_true)), 0)

        plt.savefig(s1, format='png', bbox_inches="tight")
        s1 = base64.b64encode(s1.getvalue()).decode("utf-8").replace("\n", "")
        print('<img src="data:image/png;base64,%s">' % s1)

        # if report:
        #     print('Classification Report')
        #     print(classification_report(y_test, y_pred))

    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    print_confusion_matrix(y_test, y_pred)

    # Save as a model dedicated to inference
    model.save(model_save_path, include_optimizer=False)

    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)
    print('end')

def train_log(request, device_name):
    print(device_name)
    streamer = Steamer(train_model, (device_name,))
    return StreamingHttpResponse(streamer.start(device_name))

def train_page(request, device_name):
    global p, train_flag
    train_flag = True
    d = device_name.split('-')
    if d[0] in p:
        if p[d[0]].uuid == d[1]:
            print('train', file=p[d[0]].output_file)

    return render(request, 'rtsp/train.html', {'name': d[0], 'full_name': device_name})

# --------------------------------------------
#                   test
# --------------------------------------------
def test_page(request):
    return render(request, 'rtsp/test.html')
# --------------------------------------------
#                   label
# --------------------------------------------
def annotation(request, device_name=None):
    context = {}
    context['folder_name'] = []
    context['image'] = []
    context['image_id'] = 1
    context['label_set'] = set()

    path = os.path.join(settings.BASE_DIR, 'media')
    data_folder = os.listdir(path)
    data_path = os.path.join(path, data_folder[0])
    for f in data_folder:
        dn = f.split('-')[0]
        context['folder_name'].append([dn, f])
        if f == device_name:
            data_path = os.path.join(path, f)

    if device_name != None:
        images = os.listdir(data_path)
        landmark = pd.read_csv(os.path.join(data_path, 'annotation.csv')).values.tolist()

        keypoint = []

        if request.GET:
            for key in request.GET:
                landmark[int(key)][-1] = request.GET[key]
            df = pd.DataFrame(landmark)
            df.to_csv(os.path.join(data_path, 'annotation.csv'), index=False, header=False)

        for x in images:
            if '.png' in x:
                bb = []
                for id, y in enumerate(landmark):
                    if y[0] == int(x.split(".")[0]):
                        l = np.array(y[1:43]).reshape((21,2)).T
                        min_x = l[0].min()
                        max_x = l[0].max()
                        min_y = l[1].min()
                        max_y = l[1].max()
                        width = (max_x-min_x)
                        height = (max_y-min_y)
                        bb.append([min_x-0.3*width, min_y-0.3*height, width*1.6, height*1.6, id, y[-1]])
                        if y[-1] != 'no_label':
                            context['label_set'].add(y[-1])
                            keypoint.append(y[1:])
                if len(bb) != 0:               
                    context['image'].append([x, bb])

        df = pd.DataFrame(keypoint)
        df.to_csv(os.path.join(data_path, 'keypoint.csv'), index=False, header=False)    

        context['label_set'] = list(context['label_set'])
    context['select'] = device_name
    return render(request, 'rtsp/result.html', context)

# --------------------------------------------
#               add new rtsp
# --------------------------------------------
def add_new_rtsp(request, sample_dir=None):
    global CAM
    if request.method == "POST" :
        if 'btnAddNew' in request.POST:
            new_rtsp_name = request.POST.get('new_rtsp_name')
            new_rtsp_name_url = request.POST.get('new_rtsp_url')
            RTSP.objects.create(name=new_rtsp_name, location='defualt' ,url=new_rtsp_name_url)
            
            CAM[new_rtsp_name] = LiveWebCam(new_rtsp_name_url)
    return redirect('/extra_setup/')

def clear_setup(request):
    return redirect('/extra_setup/')
# --------------------------------------------
#                main page
# --------------------------------------------
def close_device(request):
    global p
    if 'btnClose' in request.POST:
        device_name = request.POST['btnClose']
        d = device.objects.filter(name = device_name)[0]
        device_id = d.id
        id_pool.append(device_id)
        sorted(id_pool)
        # if hasattr(os.sys, 'winver'):
        #     os.kill(p[device_name].process.pid, signal.CTRL_BREAK_EVENT)
        # else:
        #     p[device_name].process.send_signal(signal.SIGTERM)
        print('deregister', file=p[device_name].output_file)
        d.delete()
        del p[device_name]
    return redirect('/')

def main_page(request, device_name=None):
    print(request.POST)
    d = device.objects.all()
    context = {}
    context['device_name'] = [ i.name for i in d]
    context['select'] = None

    if 'btnRecord' in request.GET:
        if request.GET['btnRecord'] == '1':
            context['record'] = 'record'
            print(p[device_name].process.pid)
            # passing data to process stdin by pipe
            print('record', file=p[device_name].output_file)
        elif request.GET['btnRecord'] == '-1':
            print('unrecord', file=p[device_name].output_file)
        else:
            pass

    if device_name:
        select = device.objects.filter(name = device_name)[0]
        context['select'] = select.name
        context['model'] = select.model
        context['complexity'] = 0
        context['confidence'] = int(select.confidence)

    return render(request, 'rtsp/index.html', context)
# --------------------------------------------
#                   extra setup
# --------------------------------------------
def extra_setup(request, sample_dir=None):
    rtsps = RTSP.objects.all()
    context = { 'rtsp': rtsps }
    context['cam_source'] = "--select--"
    context['cam_source_value'] = 0
    context['device_id'] = -1
    context['complexity'] = 0
    context['confidence'] = 50
    context['mediapipe_model'] = 'Hands'

    if request.method == "GET":
        # print(request.GET)
        pass

    if request.method == "POST" :
        print("--- POST ---")
        print(request.POST)
        if 'btnSave' in request.POST:
            rtsp_name = request.POST.get('rtsp_name')
            model = request.POST.get('mediapipe_model')
            complexity = int(request.POST.get('complexity'))
            confidence = int(request.POST.get('confidence'))

            if CAM.get(rtsp_name):
                
                device_id = id_pool.pop()
                if len(id_pool) == 0:
                    id_pool.append(device_id+1)
                context['device_id'] = device_id

                default_project = project.objects.all()[0]
                device_rtsp = RTSP.objects.filter(name = rtsp_name)[0]
                device_name = '{}.MP_device'.format(device_id)
                device_uuid = 'fc50c83f4251462dbafcc5e98e008b18' #uuid4().hex
                device.objects.create(  id=device_id, 
                                        name=device_name, 
                                        model=model,
                                        complexity=complexity,
                                        confidence=confidence,
                                        project=default_project, 
                                        rtsp=device_rtsp, 
                                        uuid=device_uuid)
                
                # python mpdai.py rtsp://root:pcs54784@192.168.0.33:554/live.sdp 0.mp_device hand 0 50 aaaa
                dai_path = os.path.join(settings.BASE_DIR, 'mpdai.py')
                p[device_name] = MPdevice()
                (p[device_name].pipe_r, p[device_name].pipe_w) = os.pipe()
                p[device_name].uuid = device_uuid
                p[device_name].process= subprocess.Popen(  [   'python',
                                                        dai_path,
                                                        CAM[rtsp_name].url, 
                                                        device_name, 
                                                        model, 
                                                        str(complexity), 
                                                        str(confidence),
                                                        device_uuid ],
                                                    stdin=p[device_name].pipe_r,
                                                    shell=True, 
                                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

                p[device_name].output_file = os.fdopen(p[device_name].pipe_w, 'w', buffering=1)
        try:
            rtsp_name = request.POST.get('rtsp_name')
            complexity = request.POST.get('complexity')
            confidence = request.POST.get('confidence')
            model = request.POST.get('mediapipe_model')
            # context['rtsp'] = context['rtsp'].exclude(name = rtsp_name)
            context['cam_source'] = rtsp_name
            context['cam_source_value'] = rtsp_name
            context['mediapipe_model'] = model
            context['complexity'] = complexity
            context['confidence'] = confidence
            frame_url = reverse('livecam', kwargs={'cam_name': rtsp_name, 'model': model, 'complexity':int(complexity), 'confidence':int(confidence)})
            print(frame_url)
        except:
            frame_url = static('image/NoRtspSource.png')
    else:
        frame_url = static('image/NoRtspSource.png')
    # print(context['device_id'])
    context['frame_url'] = frame_url

    return render(request, 'rtsp/extra_setup.html', context)
# --------------------------------------------
#                   stream
# --------------------------------------------
def no_livecam(request=None):
    image_data = open(os.path.join(settings.STATIC_ROOT, 'image/NoRtspSource.png'),"rb").read()
    return HttpResponse(image_data, content_type="image/png")

def gen(camera, model='Hands', complexity=0, confidence=0.5, device_name=''):
    # get keypoint tflite model 
    tflite = '.' + os.path.join( settings.MEDIA_URL, device_name,'keypoint_classifier.tflite')
    dataset = '.' + os.path.join(settings.MEDIA_URL, device_name,'keypoint.csv')

    if os.path.isfile(tflite):
        keypoint_classifier = KeyPointClassifier(model_path=tflite)
        y_dataset = np.loadtxt(dataset, delimiter=',', dtype='str', usecols=(42))
        labelencoder = LabelEncoder()
        y_dataset = labelencoder.fit_transform(y_dataset)
        gesture = labelencoder.classes_
    else:
        keypoint_classifier = None
        gesture = None

    if model == 'Hands':
        with mp_hands.Hands(
            model_complexity=complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5) as hands:
            while True:
                frame = camera.get_frame(model, hands, keypoint_classifier, gesture)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    elif model == 'Pose':
        with mp_pose.Pose(
            model_complexity=complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5) as pose:
            while True:
                frame = camera.get_frame(model, pose)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    elif model == 'Face':
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5) as face:
            while True:
                frame = camera.get_frame(model, face)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    elif model == 'Holistic':
        with mp_holistic.Holistic(
            model_complexity=complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5) as holistic:
            while True:
                frame = camera.get_frame(model, holistic)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_device(device_name):
    device_output = cv2.VideoCapture('rtsp://localhost:8554/{}'.format(device_name))
    while True:
        success, image = device_output.read()
        if not success:
            image = np.array([[[0]*640]*480]*3)
            ret, jpeg = cv2.imencode('.png', image)
            return jpeg
        height = int(960 / image.shape[1] * image.shape[0])
        image = cv2.resize(image, (960, height), interpolation = cv2.INTER_LINEAR)
        ret, png = cv2.imencode('.png', image)
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + png.tobytes() + b'\r\n\r\n')

# def device_url(request, device_name):
#     d = device.objects.filter(name= device_name)[0]    
#     return StreamingHttpResponse(gen_device("{}".format(device_name)),
#                 content_type='multipart/x-mixed-replace; boundary=frame')

def device_url(request, device_name, model, complexity=0, confidence=50):
    d = device.objects.filter(name= device_name)[0]
    confidence = float(confidence/100)
    
    if model not in models:
        return no_livecam()
    
    return StreamingHttpResponse(gen(LiveWebCam(d.rtsp.url), model, complexity, confidence, "{}-{}".format(device_name, d.uuid)),
                content_type='multipart/x-mixed-replace; boundary=frame')

def livecam_feed(request, cam_name, model, complexity, confidence):
    confidence = float(confidence/100)

    # print(cam_name, model, complexity, confidence)

    global CAM
    if not CAM.get(cam_name) or not CAM[cam_name].active or model not in models:
        return no_livecam()
    else:
        return StreamingHttpResponse(gen(LiveWebCam(CAM[cam_name].url), model, complexity, confidence),
                    content_type='multipart/x-mixed-replace; boundary=frame')


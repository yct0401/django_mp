from ast import Global
from email.policy import default
from multiprocessing import context
from select import select
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from django.templatetags.static import static
from django.urls import reverse
from django.contrib.auth.decorators import login_required

import numpy as np
import subprocess, os, signal
from rtsp.models import RTSP, device, project
from mpdevice import settings

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

models = ['Hands', 'Pose', 'Face', 'Holistic']

class LiveWebCam(object):
    def __init__(self, url):
        self.url = url
        self.cam = cv2.VideoCapture(url)
        self.active = self.cam.isOpened()
    
    def __del__(self):
        cv2.destroyAllWindows()

    def set_cam(self, url):
        self.cam = cv2.VideoCapture(url)
        self.url = url
        self.active = self.cam.isOpened()

    def get_frame(self, model, models_function):
        success, imgNp = self.cam.read()
        if not success:
            image = np.array([[[0]*640]*480]*3)
            ret, jpeg = cv2.imencode('.png', image)
            return jpeg

        image = cv2.cvtColor(cv2.flip(imgNp, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = models_function.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if model == 'Hands':
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
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
        else:
            pass
        
        height = int(960 / image.shape[1] * image.shape[0])
        image = cv2.resize(image, (960, height), interpolation = cv2.INTER_LINEAR)
        
        ret, jpeg = cv2.imencode('.png', image)
        return jpeg.tobytes()

CAM = {}
id_pool = [0]
p = {}

device.objects.all().delete()

# --------------------------------------------
#                   result
# --------------------------------------------


# --------------------------------------------
#               add new rtsp
# --------------------------------------------
for rtsp in  RTSP.objects.all():
    CAM[rtsp.name] = LiveWebCam(rtsp.url)

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
#                   device
# --------------------------------------------
def close_device(request):
    global p
    if 'btnClose' in request.POST:
        device_name = request.POST['btnClose']
        d = device.objects.filter(name = device_name)[0]
        device_id = d.id
        id_pool.append(device_id)
        sorted(id_pool)
        if hasattr(os.sys, 'winver'):
            os.kill(p[device_name].pid, signal.CTRL_BREAK_EVENT)
        else:
            p[device_name].send_signal(signal.SIGTERM)
        d.delete()       
    return redirect('/')

def main_page(request, device_name=None):
    print(request.POST)
    if 'record' in request.GET:
        print(request.GET)

    d = device.objects.all()
    context = {}
    context['device_name'] = [ i.name for i in d]
    context['select'] = None
    
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
                device_name = 'mp_device_{}'.format(device_id)
                device.objects.create(  id=device_id, 
                                        name=device_name, 
                                        model=model,
                                        complexity=complexity,
                                        confidence=confidence,
                                        project=default_project, 
                                        rtsp=device_rtsp, 
                                        screenshot_dir=None)
                # cmd = 'python mpdai.py {} {} {} {} {}'.format(CAM[rtsp_name].url, device_name, model, complexity, confidence)
                # print(cmd)
                # subprocess.run(['ls'])
                # python mpdai.py rtsp://root:pcs54784@192.168.0.33:554/live.sdp mp_device_0
                p[device_name] = subprocess.Popen(  [   'python',
                                                        'mpdai.py',
                                                        CAM[rtsp_name].url, 
                                                        device_name, 
                                                        model, 
                                                        str(complexity), 
                                                        str(confidence)],
                                                    shell=True, 
                                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
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

def gen(camera, model='Hands', complexity=0, confidence=0.5):
    if model == 'Hands':
        with mp_hands.Hands(
            model_complexity=complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5) as hands:
            while True:
                frame = camera.get_frame(model, hands)
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

def device_url(request, device_name, model, complexity=0, confidence=50):
    d = device.objects.filter(name= device_name)[0]
    confidence = float(confidence/100)
    
    if model not in models:
        return no_livecam()
    
    return StreamingHttpResponse(gen(LiveWebCam(d.rtsp.url), model, complexity, confidence),
                content_type='multipart/x-mixed-replace; boundary=frame')

def livecam_feed(request, cam_name, model, complexity, confidence):
    confidence = float(confidence/100)

    print(cam_name, model, complexity, confidence)

    global CAM
    if not CAM.get(cam_name) or not CAM[cam_name].active or model not in models:
        return no_livecam()
    else:
        return StreamingHttpResponse(gen(LiveWebCam(CAM[cam_name].url), model, complexity, confidence),
                    content_type='multipart/x-mixed-replace; boundary=frame')


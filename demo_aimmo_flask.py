import os
import sys
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from model import L2CS

import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.visualize.plotter import Plotter
from spiga.demo.analyze.analyzer import VideoAnalyzer

from flask import Flask, Response

app = Flask(__name__)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')

    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)

    args = parser.parse_args()
    return args

def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def setting():
    cudnn.enabled = True
    gpu = select_device('0', batch_size=1)
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    gaze_model = getArch('ResNet50', 90)
    print('Loading snapshot.')
    snapshot_path = args.snapshot
    saved_state_dict = torch.load(snapshot_path)
    gaze_model.load_state_dict(saved_state_dict)
    gaze_model.cuda(gpu)
    gaze_model.eval()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    faces_tracker = tr.get_tracker('RetinaSort_Res50', device=gpu)
    faces_tracker.detector.set_input_shape(height, width)

    land_processor = pr_spiga.SPIGAProcessor(dataset='wflw', gpus=[0])
    land_analyzer = VideoAnalyzer(faces_tracker, processor=land_processor)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    return cap, transformations, land_analyzer, gaze_model, height, width

def generate_frames():
    gpu = select_device('0', batch_size=1)    
    cap, transformations, land_analyzer, gaze_model, height, width = setting()
    softmax = nn.Softmax(dim=1)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            faces = land_analyzer.process_frame(frame)
            landmarks = []
            headposes = []

            if len(faces):
                bboxes = []
                for face in faces:
                    score = face.bbox[-1]

                    if score < 0.95:
                        continue

                    x_min = int(face.bbox[0])

                    if x_min < 0:
                        x_min = 0

                    y_min = int(face.bbox[1])

                    if y_min < 0:
                        y_min = 0

                    x_max = int(face.bbox[2])
                    y_max = int(face.bbox[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    bboxes.append([x_min, y_min, bbox_width, bbox_height])
                    landmarks.append(face.landmarks)
                    headposes.append(face.headpose)

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = gaze_model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(
                        pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(
                        yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0

                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                                (pitch_predicted, yaw_predicted), color=(0, 0, 255), thickness=1)

                    # cv2.putText(frame, 'FACE ID : {}'.format(face.face_id), (x_min, y_min),
                    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)

                landmarks = np.array(landmarks)
                headposes = np.array(headposes)

                plotter = Plotter()

                for bbox, landmark, headpose in zip(bboxes, landmarks, headposes):
                    x0, y0, w, h = bbox[:]

                    frame = plotter.landmarks.draw_landmarks(
                        frame, landmark, thick=1)
                    frame = plotter.hpose.draw_headpose(
                        frame, [x0, y0, x0+w, y0+h], headpose[:3], headpose[3:], euler=True)
            
            frame = np.flip(frame,axis=1)
                        
            ret, buffer = cv2.imencode('.jpg', frame)
            result_frame = buffer.tobytes()

            
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + result_frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    args = parse_args()
    app.run(host='0.0.0.0', port=35679)
import argparse
import numpy as np
import cv2
import time

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

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument('--landmark-dataset', type=str, default='wflw',
                    help='landmark dataset name')
    parser.add_argument('-t', '--tracker', type=str, default='RetinaSort',
                    choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
   
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)

    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0
  
    cap = cv2.VideoCapture(cam)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # detector = RetinaFace(gpu_id=0)
    faces_tracker = tr.get_tracker(args.tracker, device=gpu)
    faces_tracker.detector.set_input_shape(height, width)
    
    processor = pr_spiga.SPIGAProcessor(dataset=args.landmark_dataset, gpus=[int(args.gpu_id)])
    faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        success, frame = cap.read()

        while True:
            start_fps = time.time()  
           
            # faces = detector(frame)
            faces = faces_analyzer.process_frame(frame)
            landmarks = []
            headposes = []
            
            if len(faces): 
                bboxes = []
                frame_landmark = frame.copy()
                
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
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                    draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255), thickness=1)

                    cv2.putText(frame, 'FACE ID : {}'.format(face.face_id), (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1 ,cv2.LINE_AA)
                    # cv2.imwrite('/data/noah/gaze_esti.png',frame)
            
                landmarks = np.array(landmarks)
                headposes = np.array(headposes)

                plotter = Plotter()

                for bbox, landmark, headpose in zip(bboxes, landmarks, headposes):
                    x0,y0,w,h = bbox[:]
                    
                    frame = plotter.landmarks.draw_landmarks(frame, landmark)
                    frame = plotter.hpose.draw_headpose(frame, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            success,frame = cap.read()  
    
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

import os
from face_detection import RetinaFace
from model import L2CS

from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default=None, type=str)
    parser.add_argument(
        '--snapshot_landmark', help='Path of model spiga snapshot.', 
        default=None, type=str)
    parser.add_argument(
        '--input', help='path to input directory',  
        type=str)
    parser.add_argument(
        '--output', help='path to output directory',  
        type=str)
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
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
    input_path = args.input
    output_path = args.output
    
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    gaze_model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    gaze_model.load_state_dict(saved_state_dict)
    gaze_model.cuda(gpu)
    gaze_model.eval()
    
    dataset = 'wflw'
    processor = SPIGAFramework(ModelConfig(dataset, load_model_url=False, model_weights_path=args.snapshot_landmark))
    
    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    for image_name in os.listdir(input_path):
        image = cv2.imread(os.path.join(input_path, image_name))
        
        faces = detector(image)
        
        if faces is not None: 
            bboxes = []
            image_landmark = image.copy()
            
            for box, landmarks, score in faces:
                if score < .95:
                    continue
                x_min=int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min=int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max=int(box[2])
                y_max=int(box[3])
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                bboxes.append([x_min, y_min, bbox_width, bbox_height])
                
                # Crop image
                img = image[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img=transformations(im_pil)
                img  = Variable(img).cuda(gpu)
                img  = img.unsqueeze(0) 
                
                # gaze prediction
                gaze_pitch, gaze_yaw = gaze_model(img)
                
                
                pitch_predicted = softmax(gaze_pitch)
                yaw_predicted = softmax(gaze_yaw)
                
                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                
                pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                draw_gaze(x_min,y_min,bbox_width, bbox_height,image,(pitch_predicted,yaw_predicted),color=(0,0,255))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 1)


            features = processor.inference(image_landmark, bboxes)
            landmarks = np.array(features['landmarks'])
            headposes = np.array(features['headpose'])
            plotter = Plotter()
            
            for bbox, landmark, headpose in zip(bboxes, landmarks, headposes):
                x0, y0, w, h = bbox[:]

                image = plotter.landmarks.draw_landmarks(
                    image, landmark, thick=3)
                image = plotter.hpose.draw_headpose(
                    image, [x0, y0, x0+w, y0+h], headpose[:3], headpose[3:], euler=True)

        cv2.imwrite(os.path.join(output_path, image_name), image)
    

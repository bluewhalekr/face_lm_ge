import os
import copy
import shutil
import uuid
import csv
import json
import natsort
import argparse
from tqdm import tqdm

import cv2
import numpy as np

import torch.backends.cudnn as cudnn

from utils import select_device

from face_detection import RetinaFace
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

def defulat_annotation_aimmo(index, scene_id, height, width, file_id, file_name, parent_path):
    dictionary = {
                'annotations':[],
                'attributes':{},
                'metadata':{
                    'height': height,
                    'width' : width
                },
                "file_id":file_id,
                "frame_number":index,
                "scene_id": scene_id,
                'filename':file_name,
                'parent_path': parent_path,
            }
    
    return dictionary

def set_annotation_od_aimmo(idx, anno_id, points, track_id, label='head'):
    anno_bbox = {
                'id' : '{}-{}'.format(idx, anno_id),
                'type' : 'bbox',
                'track_id': track_id,
                'points' : points,
                'label' : label,
                "thumbnail_tags": [],
                'attributes':{}
            }
    
    return anno_bbox

def set_annotation_landmark_aimmo(idx, anno_id, keypoints, track_id, label='R_eye'):
    anno_landmark = {
        'id' : '{}-{}'.format(idx, anno_id),
        'type' : 'keypoint',
        'track_id' : track_id,
        'keypoints' : keypoints,
        'invisible_keys' : [],
        'label' : label
    }
    return anno_landmark

def get_max_area_det(faces, threshold=0.25):
    if faces is None:
        return None

    max_area = 0
    bbox = None
    for box, landmarks, score in faces:
        if score < threshold:
            continue
        
        x_min=0 if box[0]<0 else int(box[0])
        y_min=0 if box[1]<0 else int(box[1])                
        x_max=int(box[2])
        y_max=int(box[3])
        width = x_max - x_min
        height = y_max - y_min
        area = width*height
        
        if area > max_area:
            max_area = area
            bbox = [x_min, y_min, width, height]
    return bbox

def visualization(image, left, top, width, height, landmark, headpose):
    plotter = Plotter()
    landmark = [[0, 0] if idx not in [97, 96, 54, 82, 76] else point for idx, point in enumerate(landmark)]
    
    cv2.rectangle(image, (left, top), (left+width, top+height), (0,255,0), 1)
    image = plotter.landmarks.draw_landmarks(image, landmark, thick=3)
    image = plotter.hpose.draw_headpose(image, [left, top, left+width, top+height], headpose[:3], headpose[3:], euler=True)
    return image

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use',
        default=0, type=int)
    parser.add_argument(
        '--snapshot', help='Path of model spiga snapshot.', 
        default=None, type=str)
    parser.add_argument(
        '--input', help='path to input directory',  
        type=str)
    parser.add_argument(
        '--output', help='path to output directory',  
        type=str)
    parser.add_argument(
        '--det_threshold', help='face detection confidence score threshold',
        default=0.25, type=float)
    args = parser.parse_args()
    return args

def main(args):
    # directory setting
    input_path = args.input
    output_path = args.output
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    basename = os.path.basename(input_path)
    sub_dirs = ['image', 'bbox', 'keypoint', 'head_angle']
    
    for sub_dir_name in sub_dirs:
        os.makedirs(os.path.join(output_path, sub_dir_name), exist_ok=True)

    # model setting
    print('Loading snapshot.')
    processor = SPIGAFramework(ModelConfig(dataset_name='wflw', load_model_url=False, model_weights_path=args.snapshot), gpus=[args.gpu])    
    detector = RetinaFace(gpu_id=args.gpu)
    
    headpose_list = []    
    pre_headpose_anno = None
    for (root, dirs, files) in os.walk(input_path):
        for dir_name in dirs:
            for sub_dir_name in sub_dirs:
                os.makedirs(os.path.join(output_path, sub_dir_name, dir_name), exist_ok=True)
        
        headpose_stack = []
        scene_id = str(uuid.uuid4())    
        image_list = natsort.natsorted(filter(lambda f: f.endswith(('.jpg', '.png')), files))
        
        for idx, image_name in tqdm(enumerate(image_list)):
            image = cv2.imread(os.path.join(root, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                        
            
            # inference
            faces = detector(image)
            bbox = get_max_area_det(faces, args.det_threshold)
            
            if bbox is None:
                headpose_stack.append(
                    image_name
                )
                continue
            
            try:
                features = processor.inference(image.copy(), [bbox])
            except Exception as e:
                print('error on {} with {}'.format(image_name, e))
                headpose_stack.append(
                    image_name
                )
                continue
            
            landmark = np.array(features['landmarks'][0])
            headpose = np.array(features['headpose'][0])
        
            x0, y0 = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            
            labels_bbox = [set_annotation_od_aimmo(1, str(uuid.uuid4()), [[x0, y0], [x0+w, y0], [x0+w, y0+h], [x0,y0+h]], "1", "head")]
            labels_keypoint = [
                set_annotation_landmark_aimmo(1, str(uuid.uuid4()), {"0":[landmark[96][0],landmark[96][1]]}, "R_eye"),
                set_annotation_landmark_aimmo(2, str(uuid.uuid4()), {"0":[landmark[97][0],landmark[97][1]]}, "L_eye"),
                set_annotation_landmark_aimmo(3, str(uuid.uuid4()), {"0":[landmark[54][0],landmark[54][1]]}, "nose"),
                set_annotation_landmark_aimmo(4, str(uuid.uuid4()), {"0":[landmark[76][0],landmark[76][1]]}, "R_mouth"),
                set_annotation_landmark_aimmo(5, str(uuid.uuid4()), {"0":[landmark[82][0],landmark[82][1]]}, "L_mouth"),
            ]

            headpose_list.append(
                {
                    'path':root.replace(input_path,'/{}'.format(basename)),
                    'name':image_name,
                    'yaw':headpose[0],
                    'pitch':headpose[1],
                    'roll':headpose[2]
                }
            )
            
            # visualization
            vis_image = visualization(np.copy(image), x0, y0, w, h, landmark, headpose)
    
            # set annotation
            annotations_bbox = defulat_annotation_aimmo(idx, scene_id, image.shape[0], image.shape[1], str(uuid.uuid4()), image_name, root.replace(input_path,'/{}'.format(basename)))
            annotations_keypoint = copy.deepcopy(annotations_bbox)            
            annotations_bbox['annotations'] = labels_bbox
            annotations_keypoint['annotations'] = labels_keypoint

            # head pose interpolation
            for hp_idx, err_image in enumerate(headpose_stack):
                step = (hp_idx+1) / (len(headpose_stack)+1)
                pre_points = pre_headpose_anno
                post_points = headpose_list[-1]
                
                try:
                    headpose_list.append(
                        {
                            'path':root.replace(input_path,'/{}'.format(basename)),
                            'name':err_image,
                            'yaw':pre_points['yaw'] + (post_points['yaw']-pre_points['yaw'])*step,
                            'pitch':pre_points['pitch'] + (post_points['pitch']-pre_points['pitch'])*step,
                            'roll':pre_points['roll'] + (post_points['roll']-pre_points['roll'])*step,
                        }
                    )
                except:
                    continue         
            
            headpose_stack.clear()
            pre_headpose_anno = headpose_list[-1]
            
            
            # save 2d od annotation
            with open(os.path.join(root.replace(input_path, '{}/bbox'.format(output_path)),image_name[:-4]+'.json'),'w', encoding='utf-8') as f:
                json.dump(annotations_bbox,f, ensure_ascii=False, indent=4)

            # save landmark annotation
            with open(os.path.join(root.replace(input_path, '{}/keypoint'.format(output_path)),image_name[:-4]+'.json'),'w', encoding='utf-8') as f:
                json.dump(annotations_keypoint,f, ensure_ascii=False, indent=4)

            # save visulization result
            cv2.imwrite(os.path.join(root.replace(input_path, '{}/image'.format(output_path)),image_name), vis_image)

        # save head pose annotation
        if headpose_list:
            headpose_list = natsort.natsorted(headpose_list, key=lambda x: x['name'])            
            head_angle_csv_path = os.path.join(output_path, 'head_angle', os.path.basename(root), 'head_angle.csv')
            print('write {}'.format(head_angle_csv_path))
            with open(head_angle_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["path", "name", "yaw", "pitch", "roll"])
                writer.writeheader()

                for row in headpose_list:
                    writer.writerow(row)            
                file.close()
            headpose_list = []

if __name__ == '__main__':
    args = parse_args()
    main(args)
# Testing Script for Pose2Seg-Ski2DPose Project
# This script handles the testing and evaluation of the Pose2Seg model.

# Requirements

import cycler
import Cython
import kiwisolver
import matplotlib
import numpy
import cv2
import pkg_resources
import pycocotools
import pyparsing
import dateutil
import scipy
import six
import torch
import torchvision
import tqdm
import os
import zipfile
import random

!git clone https://github.com/liruilong940607/Pose2Seg.git

%cd Pose2Seg

!git clone https://github.com/cocodataset/cocoapi.git

%cd 'cocoapi/PythonAPI/'
!python setup.py build_ext install
%cd -

# DOWNLOAD COCO2017 DATASET

os.mkdir('/content/Pose2Seg/data/coco2017')
os.mkdir('/content/Pose2Seg/data/coco2017/annotations')

#train annotations
!wget --no-check-certificate \
    "https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json" \
    -O "/content/Pose2Seg/data/coco2017/annotations/person_keypoints_train2017_pose2seg.json"

#val annotations
!wget --no-check-certificate \
    "https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json" \
    -O "/content/Pose2Seg/data/coco2017/annotations/person_keypoints_val2017_pose2seg.json"

%cd '/content'

#val2017 images

!wget --no-check-certificate \
    "http://images.cocodataset.org/zips/val2017.zip" \
    -O "/content/Pose2Seg/data/coco2017/val2017.zip"

zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/coco2017/val2017.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/coco2017/') #Extracts the files into the /data folder
zip_ref.close()

#train2017 images

!wget --no-check-certificate \
    "http://images.cocodataset.org/zips/train2017.zip" \
    -O "/content/Pose2Seg/data/coco2017/train2017.zip"

zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/coco2017/train2017.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/coco2017/') #Extracts the files into the /data folder
zip_ref.close()


# DOWNLOAD OCHUMAN DATASET

os.mkdir('/content/Pose2Seg/data/OCHuman')
os.mkdir('/content/Pose2Seg/data/OCHuman/annotations')

#test annotations
!wget --no-check-certificate \
    "https://dl.dropboxusercontent.com/s/6we26r5298eo14a/ochuman_coco_format_test_range_0.00_1.00.json" \
    -O "/content/Pose2Seg/data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json"

#val annotations
!wget --no-check-certificate \
    "https://dl.dropboxusercontent.com/s/77n6evrrgj4d1yv/ochuman_coco_format_val_range_0.00_1.00.json" \
    -O "/content/Pose2Seg/data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"

#images
!wget --no-check-certificate \
    "https://dl.dropboxusercontent.com/s/gf4cmqodrhmmhdu/images.zip" \
    -O "/content/Pose2Seg/data/OCHuman/images.zip"


zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/OCHuman/images.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/OCHuman/') #Extracts the files into the /data folder
zip_ref.close()

!git clone https://github.com/liruilong940607/OCHumanApi
%cd OCHumanApi
!make install

!export PYTHONPATH=$PYTHONPATH:/content/Pose2Seg/

%cd '/content/Pose2Seg/'
!python cluster_pose.py

import shutil

shutil.rmtree('./imagenet_pretrain/')

#imagenet_pretrain not empty train error

#FINE CODICE COMUNE TRAIN - TEST

#download weights (15 epochs)

!wget --no-check-certificate \
    "https://www.dropbox.com/s/zcgbmdwr9kcnhl3/last.pkl?dl=1" \
    -O "/content/Pose2Seg/lastnew.pkl"


#SKI2DPOSE DATASET DOWNLOAD

os.mkdir('/content/Pose2Seg/data/Ski2DPose')
os.mkdir('/content/Pose2Seg/data/Ski2DPose/annotations')

#annotations
!wget --no-check-certificate \
    "https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_labels.json.zip" \
    -O "/content/Pose2Seg/data/Ski2DPose/annotations/ski2dpose_labels.json.zip"

zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/Ski2DPose/annotations/ski2dpose_labels.json.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/Ski2DPose/annotations/') #Extracts the files into the /data folder
zip_ref.close()

#orginal images
!wget --no-check-certificate \
    "https://dl.dropboxusercontent.com/s/p9wbbs9f6d4y8q2/ski2dpose_images_jpg.zip" \
    -O "/content/Pose2Seg/data/Ski2DPose/ski2dpose_images_jpg.zip"


zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/Ski2DPose/ski2dpose_images_jpg.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/Ski2DPose/') #Extracts the files into the /data folder
zip_ref.close()

#images not in subfolder
!wget --no-check-certificate \
    "https://www.dropbox.com/scl/fi/7q0ldzpzk8khj8j63i0mg/ski2dpose_coco_format.zip?rlkey=600ej2zdsp5jre9v5v0w53hus&dl=1" \
    -O "/content/Pose2Seg/data/Ski2DPose/ski2dpose_imagesall.zip"


zip_ref = zipfile.ZipFile('/content/Pose2Seg/data/Ski2DPose/ski2dpose_imagesall.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/content/Pose2Seg/data/Ski2DPose/') #Extracts the files into the /data folder
zip_ref.close()

# Dataloader Ski2dpose

import sys
import os
import json
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from google.colab.patches import cv2_imshow

from shapely.geometry import Polygon

class SkiDataset(Dataset):
    '''ALPINE SKII DATASET'''

    # BGR channel-wise mean and std
    all_mean = torch.Tensor([190.24553031, 176.98437134, 170.87045832]) / 255
    all_std  = torch.Tensor([ 36.57356531,  35.29007466,  36.28703238]) / 255
    train_mean = torch.Tensor([190.37117484, 176.86400202, 170.65409075]) / 255
    train_std  = torch.Tensor([ 36.56829177,  35.27981661,  36.19375109]) / 255

    coco_joints = {
        'head' : 'nose', #0
        'neck' : 'neck', #1
        'shoulder_right' : 'right_shoulder', #2
        'elbow_right' : 'right_elbow', #3
        'hand_right' : 'right_hand', #4
        'pole_basket_right' : '', #5 (Sci)
        'shoulder_left' : 'left_shoulder', #6
        'elbow_left' : 'left_elbow', #7
        'hand_left' : 'left_hand', #8
        'pole_basket_left' : '', #9 (Sci)
        'hip_right' : 'right_hip', #10
        'knee_right' : 'right_knee', #11
        'ankle_right' : 'right_ankle', #12
        'hip_left' : 'left_hip', #13
        'knee_left' : 'left_knee', #14
        'ankle_left' : 'left_ankle', #15
        'ski_tip_right' : '', #16 (Sci)
    }

    coco_bones = [[0,1], [1,2], [2,3], [3,4], [1,6], [6,7], [7,8], [2,10], [10,11], [11,12], [6,13], [13,14], [14,15]]


    # VideoIDs and SplitIDs used for validation
    val_splits = [('5UHRvqx1iuQ', '0'), ('5UHRvqx1iuQ', '1'),
                  ('oKQFABiOTw8', '0'), ('oKQFABiOTw8', '1'), ('oKQFABiOTw8', '2'),
                  ('qxfgw1Kd98A', '0'), ('qxfgw1Kd98A', '1'),
                  ('uLW74013Wp0', '0'), ('uLW74013Wp0', '1'),
                  ('zW1bF2PsB0M', '0'), ('zW1bF2PsB0M', '1')]


    def __init__(self, imgs_dir, label_path, img_extension='jpg', mode='all', img_size=(1920,1080),
                 normalize=True, in_pixels=True, return_info=False):
        '''
        Create a Ski2DPose dataset loading train or validation images.

        Args:
            :imgs_dir: Root directory where images are saved
            :label_path: Path to label JSON file
            :img_extension: Image format extension depending on downloaded version. One of {'png', 'jpg', 'webp'}
            :mode: Specify which partition to load. One of {'train', 'val', 'all'}
            :img_size: Size of images to return
            :normalize: Set to True to normalize images
            :in_pixels: Set to True to scale annotations to pixels
            :return_info: Set to True to include image names when getting items

            :img: Input image (in pixels)
            :an: Annotation positions tensor (in pixels)
            :vis: Visibility flag tensor
            :info: (video_id, split_id, img_id, frame_idx) tuple
        '''
        self.imgs_dir = imgs_dir
        self.img_extension = img_extension
        self.mode = mode
        self.img_size = img_size
        self.normalize = normalize
        self.in_pixels = in_pixels
        self.return_info = return_info

        assert mode in ['train', 'val', 'all'], 'Please select a valid mode.'
        self.mean = self.all_mean if self.mode == 'all' else self.train_mean
        self.std = self.all_std if self.mode == 'all' else self.train_std

        # Load annotations
        with open(label_path) as f:
            self.labels = json.load(f)

        # Check if all images exist and index them
        self.index_list = []

        for video_id, all_splits in self.labels.items():
            for split_id, split in all_splits.items():
                for img_id, img_labels in split.items():


                    img_path = os.path.join(imgs_dir, video_id, split_id, '{}.{}'.format(img_id, img_extension))
                    if os.path.exists(img_path):
                        if ((mode == 'all') or
                            (mode == 'train' and (video_id, split_id) not in self.val_splits) or
                            (mode == 'val' and (video_id, split_id) in self.val_splits)):
                            self.index_list.append((video_id, split_id, img_id))
                    else:
                        print('Did not find image {}/{}/{}.{}'.format(video_id, split_id, img_id, img_extension))

    def __len__(self):
        return len(self.index_list) #Returns the number of samples in the dataset

    def __getitem__(self, index):

        # Load annotations
        video_id, split_id, img_id = self.index_list[index]
        annotation = self.labels[video_id][split_id][img_id]['annotation']
        frame_idx  = self.labels[video_id][split_id][img_id]['frame_idx']

        an = torch.Tensor(annotation)[:,:2]
        vis = torch.LongTensor(annotation)[:,2]

        img_path = os.path.join(self.imgs_dir, video_id, split_id, '{}.{}'.format(img_id, self.img_extension))
        img = cv2.imread(img_path) # (H x W x C) BGR

        img = torch.from_numpy(img)

        if self.normalize:
            img = ((img / 255) - self.mean) / self.std

        if self.in_pixels:
            an *= torch.Tensor([img.shape[1], img.shape[0]])

        # Create index to remove nodes 5, 9 (ski poles) and 16 to 24 (ski elements)
        indice = [i for i in range(16) if i not in [5, 9]]
        keypoints = an[indice]

        # Get x, y coordinates of keypoints
        kp_x = [pt[0] for pt in keypoints[:, :2]]
        kp_y = [pt[1] for pt in keypoints[:, :2]]

        # Calculate min and max values of x, y coordinates
        min_x = torch.min(keypoints[:, 0])
        max_x = torch.max(keypoints[:, 0])
        min_y = torch.min(keypoints[:, 1])
        max_y = torch.max(keypoints[:, 1])

        # Add padding to bounding box values
        padding = 30
        min_x, min_y, max_x, max_y = min_x - padding, min_y - padding, max_x + padding, max_y + padding

        # Define bbox min and max coords
        bbox = [min_x, min_y, max_x, max_y]
        bbox = np.array(bbox)
        bbox = bbox.tolist()

        # Get keypoints coords
        keypoint_coords = [(float(pt[0]), float(pt[1])) for pt in keypoints]

        # Create polygon object from these coordinates
        polygon = Polygon(keypoint_coords)
        polygon_points = list(polygon.exterior.coords)

        # List of coordinates
        segmentation = [float(coord) for point in polygon_points for coord in point]
        segmx = [segmentation]

        if self.return_info:
            return img, an, vis, bbox, segmx, (video_id, split_id, img_id, frame_idx)

        return img, an, vis, bbox, segmx

    def annotate_img(self, img, an, vis, info=None):

        width, height = img.shape[1], img.shape[0]
        img = img.numpy()
        # Scale based on head-foot distance
        scale = torch.norm(an[0] - an[15]) / height
        img_an = img.copy()
        # Draw all bones
        for bone_from, bone_to in self.coco_bones:
            x_from, y_from = int(an[bone_from][0].item()), int(an[bone_from][1].item())
            x_to, y_to = int(an[bone_to][0].item()), int(an[bone_to][1].item())
            cv2.line(img_an, (x_from, y_from), (x_to, y_to), (0,255,0), int(max(2,5*scale)))
        # Draw all joints
        for (x,y), flag in zip(an, vis):
            color = (0,0,255) if flag == 1 else (255,0,0)
            cv2.circle(img_an, (int(x.item()), int(y.item())), int(max(2,14*scale)), color, -1)

        if info is not None:
            text = 'Image {}, frame {}.'.format(info[2], info[3])
            cv2.putText(img_an, text, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5*(width/1920),
                        (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(img_an, text, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5*(width/1920),
                        (255,255,255), 2, cv2.LINE_AA)
        return img_an

        # End Skidataset class ------------------

def determine_image_format():
    base_path = "/content/Pose2Seg/data/Ski2DPose/"
    formats = ['png', 'webp', 'jpg']

    for img_format in formats:
        img_dir = Path(base_path + f'Images_{img_format}')
        if img_dir.is_dir():
            print(f'Found image directory {img_dir}, using {img_format} format.')
            return img_format, img_dir

    raise FileNotFoundError('Image directory not found, please ensure one of the following directories exists: ' + ', '.join(f'Images_{ext}' for ext in formats))

if __name__ == '__main__':
    print('Dataloader for 2D alpine ski dataset')

    label_path = '/content/Pose2Seg/data/Ski2DPose/annotations/ski2dpose_labels.json'
    img_extension, imgs_dir = determine_image_format()

    ski_dataset = SkiDataset(imgs_dir=imgs_dir, label_path=label_path, img_extension=img_extension,
                         img_size=(1920,1080), mode='all', normalize=False, in_pixels=True, return_info=True)
    print('Number of images: {}'.format(len(ski_dataset)))

    # Create COCO annotations
    coco_annotations = {
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": SkiDataset.coco_joints,
                "skeleton": SkiDataset.coco_bones
            }
        ],
        "images": [],
        "annotations": []
    }

    image_id_counter = 0

    # Process each image and annotation in the dataset
    for img, an, vis, bbox, segmx, info in ski_dataset:
        # Define COCO image entry
        coco_img_entry = {
            "id": image_id_counter,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": f"{info[2]}.{img_extension}",
            "license": 1,
        }

        # Set visibility of ski pole keypoints (5, 9) and other ski elements (16-24) to 0
        vis[5] = vis[9] = 0
        for i in range(16, len(vis)):
            vis[i] = 0

        keypoints_coco = []

        # Loop through first 17 elements and set vis from 1 to 2 (coco requirements)
        for i in range(17):
            point, v = an[i], vis[i]
            v_coco = 2 if v == 1 else 0

            if i == 5 or i == 9 or i == 16:
              keypoints_coco.extend([0, 0, 0])
            else:
              keypoints_coco.extend([point[0].item(), point[1].item(), v_coco])

        # Define COCO annotation entry
        coco_annotation_entry = {
            "id": image_id_counter,
            "image_id": image_id_counter,
            "category_id": 1,
            "keypoints": keypoints_coco,
            "num_keypoints": int(vis.sum()),
            "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
            "bbox": bbox,
            "iscrowd": 0,
            "segmentation": segmx
        }

        image_id_counter += 1
        coco_annotations["images"].append(coco_img_entry)
        coco_annotations["annotations"].append(coco_annotation_entry)

        # Annotate and display the image with joints
        img_an = ski_dataset.annotate_img(img, an, vis, info)
        #cv2_imshow(cv2.resize(img_an, (960, 540))) if image_id_counter < 6 else None

        # Random index
        random_indices = random.sample(range(len(ski_dataset)), 30)
        # Draw bbox on img
        img_with_bbox = img_an.copy()
        x1, y1, x2, y2 = map(int, bbox)  # Extract coords
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2_imshow(img_with_bbox) if image_id_counter in random_indices else None

    # --------------------------------------------------------

    # Save COCO annotations to a JSON file
    coco_output_path = '/content/Pose2Seg/data/Ski2DPose/annotations/ski2dpose_labels_coco.json'
    with open(coco_output_path, 'w') as coco_file:
        json.dump(coco_annotations, coco_file)

    # Verify the saved COCO annotations
    print(f"COCO-formatted annotations saved at: {coco_output_path}")

#TEST POSE2SEG ON VAL SET

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

def test(model, dataset='cocoVal', logger=print):
    if dataset == 'OCHumanVal':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = './data/coco2017/val2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'
    elif dataset == 'Ski2DPose':
        ImageRoot = './data/Ski2DPose/images'
        AnnoFile = './data/Ski2DPose/annotations/ski2dpose_labels_coco.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)

    model.eval()

    results_segm = []
    imgIds = []

    # Random index for visualize images
    random_indices = random.sample(range(len(datainfos)), 30)

    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']

        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])

        output = model([img], [gt_kpts], [gt_masks]) #OUTPUT MODEL

        # Convert output in mask
        pred_masks = output[0]

        # View images, keypoints and pred mask
        if i in random_indices:
            fig, axs = plt.subplots(1, 3, figsize=(20, 20))
            axs[0].imshow(gt_masks[0])  # assuming gt_masks is a list of masks
            axs[0].set_title('Ground Truth Mask')

            # Draw keypoints
            for kpt in gt_kpts[0]:
                x, y, _ = kpt
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Cerchio verde
            axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[1].set_title('Original Image w/ Keypoints')

            axs[2].imshow(pred_masks[0])
            axs[2].set_title('Predicted Mask')
            plt.show()

        for mask in pred_masks:
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
        imgIds.append(image_id)


    def do_eval_coco(image_ids, coco, results, flag):
        from pycocotools.cocoeval import COCOeval
        assert flag in ['bbox', 'segm', 'keypoints']
        # Evaluate
        coco_results = coco.loadRes(results)
        cocoEval = COCOeval(coco, coco_results, flag)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval

    cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    logger('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    _str = '[segm_score] %s '%dataset
    for value in cocoEval.stats.tolist():
        _str += '%.3f '%value
    logger(_str)

# Argparse variables
weights = '/content/Pose2Seg/lastnew.pkl'
coco = True
OCHuman = True
Ski2DPose = True

print('===========> LOADING MODEL <===========')
model = Pose2Seg().cuda()
model.init(weights)

print('===========>   TESTING    <===========')
if Ski2DPose:
    print('----------------------------------')
    print('TEST SKI2DPOSE')
    test(model, dataset='Ski2DPose')
if coco:
    print('----------------------------------')
    print('TEST COCO VAL')
    test(model, dataset='cocoVal')
if OCHuman:
    print('----------------------------------')
    print('TEST OCHUMAN VAL')
    test(model, dataset='OCHumanVal')
    print('----------------------------------')
    print('TEST OCHUMAN TEST')
    test(model, dataset='OCHumanTest')


#!python train.py

'''from google.colab import files

files.download('/content/Pose2Seg/snapshot/release_base_2024-01-25_11:17:58/last.pkl')'''

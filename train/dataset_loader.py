import numpy as np
import os
import random

from PIL import Image
import torch
import torchvision

from torch.utils.data import Dataset
import glob


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class SegmentationDataset(Dataset):
    
    def __init__(self, root, subset,
                img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None):


        # print(img_path)
        self.images_root = f'{root}/{img_path}/{subset}'
        self.labels_root = f'{root}/{label_path}/{subset}'
        self.image_paths = glob.glob(f'{self.images_root}/{pattern}')
        self.label_paths = [ img.replace(self.images_root, self.labels_root).replace(img_suffix, label_suffix) for img in self.image_paths  ]
        if "idd" in root:
            self.image_paths = self.image_paths[:4000]
            self.label_paths = self.label_paths[:4000]
        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]

        self.file_path = file_path
        self.transform = transform
        self.relabel = Relabel(255, self.num_classes) if transform != None else None


    def __getitem__(self, index):

        filename = self.image_paths[index]
        filenameGt = self.label_paths[index]


        with Image.open(filename) as f:
            image = f.convert('RGB')

        if self.mode == 'labeled':
            with Image.open(filenameGt) as f:
                label = f.convert('P')
        else:
            label = image

        # print(image.size, label.size)
        if self.transform !=None:
            image, label = self.transform(image, label)

        if self.d_idx == 'NYUv2_s': ## Wrap around the void class
            label = label-1
            label[label<0] = 255

        if self.relabel != None and self.mode == 'labeled':
            label = self.relabel(label)


        if self.mode == 'unlabeled':
            return image
        else:
            return image, label


    def __len__(self):
        return len(self.image_paths)


class CityscapesDataset(SegmentationDataset):

    num_classes = 19
    label_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    color_map = np.array([
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32]
    ], dtype=np.uint8)


    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled'):
        self.d_idx = 'CS'
        self.mode = mode
        super(CityscapesDataset, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelTrainIds.png', transform=transform, file_path=file_path, num_images=num_images)

class MapillaryDataset(SegmentationDataset):

    num_classes = 25
    label_names = ['pole', 'street light', 'billboard', 'traffic light', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic sign front', 'traffic sign back', 'road', 'sidewalk', 'curb', 'fence', 'wall', 'building', 'person', 'motorcyclist', 'bicyclist', 'sky', 'vegetation', 'terrain', 'general marking', 'crosswalk zebra']

    color_map = np.array([
        [153,153,153],# pole
        [210,170,100],# street light
        [220,220,220],# billboard
        [250,170, 30],# traffic light
        [  0,  0,142],# car
        [  0,  0, 70],# truck
        [119, 11, 32],# bicycle
        [  0,  0,230],# motorcycle
        [  0, 60,100],# bus
        [220,220,  0],# traffic sign front 
        [192,192,192],# traffic sign back
        [128, 64,128],# road
        [244, 35,232],# sidewalk
        [196,196,196],# curb
        [190,153,153],# fence
        [102,102,156],# wall
        [ 70, 70, 70],# building
        [220, 20, 60],# person
        [255,  0,100],# motorcyclist
        [255,  0,  0],# bicyclist
        [ 70,130,180],# sky
        [107,142, 35],# vegetation
        [152,251,152],# terrain
        [255,255,255],# general marking
        [200,128,128] # crosswalk zebra 
    ], dtype=np.uint8)


    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled'):
        self.d_idx = 'MAP'
        self.mode = mode
        super(MapillaryDataset, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelIds.png', transform=transform, file_path=file_path, num_images=num_images)
        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class ADE20KDataset(SegmentationDataset):

    num_classes = 50
    label_names = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window', 'grass', 'cabinet', 'sidewalk', 'person', 'ground', 'door', 'table', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'sea', 'mirror', 'carpet', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'closet', 'lamp', 'tub', 'cushion', 'box', 'pillar', 'sign', 'sink', 'path', 'stairs', 'pillow', 'stairway', 'light', 'street light', 'pole', 'escalator', 'step']

    color_map = np.array([
        [120,120,120], # wall
        [180,120,120], # building
        [  6,230,230], # sky
        [ 80, 50, 50], # floor
        [  4,200,  3], # tree
        [120,120, 80], # ceiling
        [140,140,140], # road
        [204,  5,255], # bed
        [230,230,230], # window
        [  4,250,  7], # grass
        [224,  5,255], # cabinet
        [235,255,  7], # sidewalk
        [150,  5, 61], # person
        [120,120, 70], # ground
        [  8,255, 51], # door
        [255,  6, 82], # table
        [204,255,  4], # plant
        [255, 51,  7], # curtain
        [204, 70,  3], # chair
        [  0,102,200], # car
        [ 61,230,250], # water
        [255,  6, 51], # painting
        [ 11,102,255], # sofa
        [255,  7, 71], # shelf
        [  9,  7,230], # sea
        [220,220,220], # mirror
        [255,  9, 92], # carpet
        [112,  9,255], # field
        [  8,255,214], # armchair
        [  7,255,224], # seat
        [255,184,  6], # fence
        [ 10,255, 71], # desk
        [255, 41, 10], # rock
        [  7,255,255], # closet
        [224,255,  8], # lamp
        [102,  8,255], # tub
        [255,194,  7], # cushion
        [  0,255, 20], # box
        [255,  8, 41], # pillar
        [255,  5,153], # sign
        [  0,163,255], # sink
        [255, 31,  0], # path
        [255,224,  0], # stairs
        [  0,235,255], # pillow
        [ 31,  0,255], # stairway
        [255,173,  0], # light
        [  0, 71,255], # street light
        [ 51,  0,255], # pole
        [  0,255,163], # escalator
        [255,  0,143]  # step
    ], dtype=np.uint8)


    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled'):
        self.d_idx = 'ADE'
        self.mode = mode
        super(ADE20KDataset, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelIds.png', transform=transform, file_path=file_path, num_images=num_images)
        self.relabel = Relabel(255, self.num_classes) if transform != None else None


class ANL4Transform(object):


    def __call__(self, image, label):
        indices = label >= 30
        label[indices] = 255
        return image, label

        


class ANUEDatasetL4(SegmentationDataset):

    num_classes = 30
    label_names = ['road', 'parking', 'drivable fallback', 'sidewalk',  'non-drivable fallback', 'person', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan',  'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'fallback background']

    color_map = np.array([[128, 64, 128], [250, 170, 160], [81, 0, 81], [244, 35, 232], [152, 251, 152], [220, 20, 60], [246, 198, 145], [255, 0, 0], [0, 0, 230], [119, 11, 32], [255, 204, 54], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [136, 143, 153], [220, 190, 40], [102, 102, 156], [190, 153, 153], [180, 165, 180], [174, 64, 67], [220, 220, 0], [250, 170, 30], [153, 153, 153], [0, 0, 0], [70, 70, 70], [150, 100, 100], [107, 142, 35], [70, 130, 180], [169, 187, 214]], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None):
        self.d_idx = 'ANUE'
        super(ANUEDatasetL4, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel4Ids.png', transform=transform, file_path=file_path, num_images=num_images)



class IDD_Dataset(SegmentationDataset):

    num_classes = 26
    label_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

    color_map   = np.array([
        [128, 64, 128], #road
        [ 81,  0, 81], #drivable fallback
        [244, 35, 232], #sidewalk
        [152, 251, 152], #nondrivable fallback
        [220, 20, 60], #pedestrian
        [255, 0, 0],  #rider
        [0, 0, 230], #motorcycle
        [119, 11, 32], #bicycle
        [255, 204, 54], #autorickshaw
        [0, 0, 142], #car
        [0, 0, 70], #truck
        [0, 60, 100], #bus
        [136, 143, 153], #vehicle fallback
        [220, 190, 40], #curb
        [102, 102, 156], #wall
        [190, 153, 153], #fence
        [180, 165, 180], #guard rail
        [174, 64, 67], #billboard
        [220, 220, 0], #traffic sign
        [250, 170, 30], #traffic light
        [153, 153, 153], #pole
        [169, 187, 214], #obs-str-bar-fallback
        [70, 70, 70], #building
        [150, 120, 90], #bridge
        [107, 142, 35], #vegetation
        [70, 130, 180] #sky
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled'):
        self.d_idx = 'IDD'
        self.mode = mode
        super().__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images)
        
        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class IDD20KDataset(SegmentationDataset):

    num_classes = 26
    label_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

    color_map   = np.array([
        [128, 64, 128], #road
        [ 81,  0, 81], #drivable fallback
        [244, 35, 232], #sidewalk
        [152, 251, 152], #nondrivable fallback
        [220, 20, 60], #pedestrian
        [255, 0, 0],  #rider
        [0, 0, 230], #motorcycle
        [119, 11, 32], #bicycle
        [255, 204, 54], #autorickshaw
        [0, 0, 142], #car
        [0, 0, 70], #truck
        [0, 60, 100], #bus
        [136, 143, 153], #vehicle fallback
        [220, 190, 40], #curb
        [102, 102, 156], #wall
        [190, 153, 153], #fence
        [180, 165, 180], #guard rail
        [174, 64, 67], #billboard
        [220, 220, 0], #traffic sign
        [250, 170, 30], #traffic light
        [153, 153, 153], #pole
        [169, 187, 214], #obs-str-bar-fallback
        [70, 70, 70], #building
        [150, 120, 90], #bridge
        [107, 142, 35], #vegetation
        [70, 130, 180] #sky
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled'):
        self.d_idx = 'IDD20K'
        self.mode = mode
        super().__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*/*',
                img_suffix = '_leftImg8bit.jpg' , label_suffix='_gtFine_labellevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images)
        
        self.relabel = Relabel(255, self.num_classes) if transform != None else None


class CamVid(SegmentationDataset):

    num_classes = 11
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'CVD'
        self.mode = mode


        self.images_root = f"{root}/{subset}/"
        self.labels_root = f"{root}/{subset}annot/"
        

        self.image_paths = glob.glob(f'{self.images_root}/*.png')
        self.label_paths = glob.glob(f'{self.labels_root}/*.png')

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class SunRGB(SegmentationDataset):

    num_classes = 37
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'SUN'
        self.mode = mode

        listname = f"{root}/{subset}37.txt"

        with open(listname , 'r') as fh:
            self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        with open(listname , 'r') as fh:
            self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class NYUv2_seg(SegmentationDataset):

    num_classes = 13
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'NYU_s'
        self.mode = mode

        # listname = f"{root}/{subset}13.txt"

        images = os.listdir(os.path.join(root , subset , 'images'))
        labels = os.listdir(os.path.join(root , subset , 'labels'))

        self.image_paths = [f"{root}/{subset}/images/"+im_id for im_id in images]
        self.label_paths = [f"{root}/{subset}/labels/"+lb_id for lb_id in labels]

        # with open(listname , 'r') as fh:
        #     self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        # with open(listname , 'r') as fh:
        #     self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None


def colorize(img, color, fallback_color=[0,0,0]): 
    img = np.array(img)
    W,H = img.shape
    view = np.tile(np.array(fallback_color, dtype = np.uint8), (W,H, 1) )
    for i, c in enumerate(color):
        indices = (img == i)
        view[indices] = c
    return view


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    

    def show_data(ds):
        print(len(ds))
        i = random.randrange(len(ds))
        img, gt = ds[i]
        color_gt = colorize(gt, ds.color_map)
        print(img.size,color_gt.shape)
        plt.imshow(img)
        plt.imshow(color_gt, alpha=0.25)
        plt.show()


    # cs = CityscapesDataset('/ssd_scratch/cvit/girish.varma/dataset/cityscapes')
    # show_data(cs)

    # an = ANUEDataset('/ssd_scratch/cvit/girish.varma/dataset/anue')
    # show_data(an)

    # bd = BDDataset('/ssd_scratch/cvit/girish.varma/dataset/bdd100k')
    # show_data(bd)

    # mv = MVDataset('/ssd_scratch/cvit/girish.varma/dataset/mvd')
    # show_data(mv)

import torch
import requests
import os
import zipfile
from io import BytesIO
from urllib.parse import urlparse
from videostitching.yolo_functions import select_device
from models.yolo import attempt_load
from videostitching.yolo_functions import non_max_suppression, scale_coords, xyxy2xywh, from_yolo_toxy
import numpy as np
import cv2
import logging

"""
def set_logging(rank=-1, verbose=True):
            logging.basicConfig(
                format="%(message)s",
                level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)
                
@torch.no_grad()
def load_weights_model(wpath, device='', half=False, yolo =True):
    set_logging()
    device = select_device(device)

    half &= device.type != 'cpu'
    w = str(wpath[0] if isinstance(wpath, list) else wpath)
    if yolo:
        if 'torchscript' in w:
            model = torch.jit.load(w)
        else:
            model = attempt_load(wpath, map_location=device)
        
    else:
        model = torch.jit.load(w)

    if half:
        model.half()  # to FP16

    return [model, device, half]
"""

def draw_frame(img, bbbox, dictlabels = None, default_color = (255,255,255)):
    imgc = img.copy()
    for i in range(len(bbbox)):
        x1,x2,y1,y2 = bbbox[i]

        widhtx = abs(x1 - x2)
        heighty = abs(y1 - y2)

        start_point = (x1, y1)
        end_point = (x2,y2)
        if dictlabels is not None:
            color = dictlabels[i]['color']
            label = dictlabels[i]['label']
        else:
            label = ''
            color = default_color

        thickness = 4
        xtxt = x1 if x1 < x2 else x2
        ytxt = y1 if y1 < y2 else y2
        imgc = cv2.rectangle(imgc, start_point, end_point, color, thickness)
        if label != '':

            imgc = cv2.rectangle(imgc, (xtxt,ytxt), (xtxt + int(widhtx*0.8), ytxt - int(heighty*.2)), color, -1)
            
            imgc = cv2.putText(img=imgc, text=label,org=(xtxt + int(abs(x1-x2)/15),
                                                            ytxt - int(abs(y1-y2)/20)), 
                                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1*((heighty)/200), color=(255,255,255), thickness=2)
            
    return imgc   

def download_weigths(url):

    a = urlparse(url)

    if not os.path.exists(os.path.basename(a.path)):
        req = requests.get(url)

        with zipfile.ZipFile(BytesIO(req.content)) as zipobject:
            zipobject.extractall('weigths')

    else:
        with zipfile.ZipFile(os.path.basename(a.path)) as zipobject:
            zipobject.extractall('weigths')

    return a

def readweigths_frompath(weigth_path, suffix='.pt'):
    if weigth_path.startswith('http'):

        a = download_weigths(weigth_path)
        newpathtomodel = os.path.join('weigths')  # remove.zip
        fileinfolder = [i for i in os.listdir(newpathtomodel) if i.endswith(suffix)]

        if len(fileinfolder) == 1:
            wp = fileinfolder[0]
            wp = os.path.join(newpathtomodel, wp)
        else:
            raise ValueError("there is no weights files")

    elif weigth_path.endswith(suffix):
        wp = weigth_path
    else:
        raise ValueError("there is no weights files with {} extension".format(suffix))

    return wp


class DetectCC(object):

    @property
    def imgscrop_detections(self):

        clippedimgs = []
        if type(self.bb_coords) is list:
            for i in range(len(self.bb_coords)):
                l, r, t, b = self.bb_coords[i]
                clippedimgs.append(self.image[t:b, l:r, ])

        return clippedimgs

    @property
    def detector_model(self):
        return self._model

    def _check_image(self):

        imgc = self.image.copy()

        if len(imgc.shape) == 3:
            imgc = np.expand_dims(imgc, axis=0)

        if imgc.shape[3] == 3:
            if (not imgc.shape[2] == self.inputshape[0]) and not (imgc.shape[3] == self.inputshape[1]):
                imgc = cv2.resize(imgc[0], self.inputshape, interpolation=cv2.INTER_AREA)
                imgc = np.expand_dims(imgc, axis=0)
            imgc = imgc.swapaxes(3, 2).swapaxes(2, 1)
        else:
            imgc = imgc.swapaxes(1, 2).swapaxes(2, 3)
            if (not imgc.shape[2] == self.inputshape[0]) and not (imgc.shape[3] == self.inputshape[1]):
                imgc = cv2.resize(imgc[0], self.inputshape, interpolation=cv2.INTER_AREA)
            imgc = imgc.swapaxes(3, 2).swapaxes(2, 1)

        self._imgtopredict = imgc

    def _check_weigths_path(self):
        self.weigths_path = readweigths_frompath(self.weigths_path)

    def _set_yolomodel(self):

        #set_logging()
        self.device = select_device('')
        self._model = torch.load(self.weigths_path) #load_weights_model(self.weigths_path) #torch.load(self.weigths_path) #
        #self._model, _, _ = load_weights_model(self.weigths_path)  # torch.load(self.weigths_path) #

        self._model.to(self.device)
        ## TODO: automaticaly figure out input model shape
        self.inputshape = (480, 480)

    def predict(self, image):

        self.image = image
        self._check_image()
        img = torch.from_numpy(self._imgtopredict).to(self.device)
        img = img / 255.
        self.bounding_box = self.detector_model(img, augment=False)

    def get_boundaryboxes(self, conf_thres=0.60, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):

        pred = non_max_suppression(self.bounding_box[0], conf_thres, iou_thres, classes,
                                   agnostic_nms, max_det=max_det)
        xyxylist = []

        for i, det in enumerate(pred):
            s, im0 = '', np.squeeze(self.image).copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                
                det[:, :4] = scale_coords(self._imgtopredict.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    # Rescale boxes from img_size to im0 size
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # xywh = [torch.tensor(xyxy).tolist(), xywh, conf.tolist()]
                    m = [0]
                    
                    for i in range(len(xywh)):
                        m.append(xywh[i])

                    l, r, t, b = from_yolo_toxy(m, (im0.shape[0],
                                                    im0.shape[1]))

                    xyxylist.append([l, r, t, b])

        self.bb_coords = xyxylist
        return self.bb_coords

    def __init__(self, weigths_path=None, modelname="yolo") -> None:
        self.bb_coords = None
        self.weigths_path = weigths_path
        self._check_weigths_path()

        self._set_yolomodel()

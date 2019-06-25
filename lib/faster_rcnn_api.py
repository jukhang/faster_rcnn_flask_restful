import os
import cv2
import sys
import numpy as np
from lib.blob import im_list_to_blob

#im_file = os.path.join('/data/wxh/tf-faster-rcnn/data/demo/000004.jpg')
#im = cv2.imread(im_file)

def get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    TEST_SCALES = (600,)
    TEST_MAX_SIZE = 1000

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1000:
            im_scale = float(1000) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = get_image_blob(im)
    
    return blobs, im_scale_factors


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    
    return boxes


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
                          
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    
    result_list = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255, 0, 255),thickness=2)
        cv2.putText(im,class_name,(bbox[0],int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        
        key_list = ['class_name', 'position', 'score']
        # return result
        value_list = [class_name, str(bbox), str(score)]
        result = dict(zip(key_list, value_list))
        result_list.append(result)

    return result_list


def im_detect(im, im_file):
    blobs, im_scales = get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    input0 = blobs['data'].tolist()
    input1 = blobs['im_info'].tolist()
    parms = {"signature_name": "tf_faster_rcnn_cls", 
             "inputs":{
                      'input0': input0,
                      'input1' : input1
                     }
            }
    
    from urllib import parse
    import requests
    import json
    headers = {'Content-Type': 'application/json'}
    url = 'http://localhost:8501/v1/models/saved_model:predict'
    
    u = requests.post(url , headers=headers, data=json.dumps(parms)).json()
    u = u['outputs']
    _nosess = np.array(u['output0'])
    scores = np.array(u['output1'])
    bbox_pred = np.array(u['output2'])
    rois = np.array(u['output3'])
    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    
    if True:
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    
    ###### boxes to pred_boxes
    boxes = pred_boxes

    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    
    from lib.nms_wrapper import nms
    result_list = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        result = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        if result is not None:
            #print(result)
            result_list.append(result)
        
    pwd = os.path.join('/data/wxh/www/result/', im_file)
    cv2.imwrite(pwd, im)
    return result_list

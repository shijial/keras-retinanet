"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from keras_retinanet.utils.image import preprocess_image, resize_image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os
import cv2


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    one_x = boxA[0]
    one_y = boxA[1]
    one_w = boxA[2] - boxA[0]
    one_h = boxA[3] - boxA[1]

    two_x = boxB[0]
    two_y = boxB[1]
    two_w = boxB[2]
    two_h = boxB[3]

    if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        calcIOU = inter_square / union_square * 1.0
    else:
        calcIOU = 0

    return calcIOU

def evaluate_voc(generator, model, threshold=0.05):
    # start collecting results
    # results = []
    # image_ids = []
    results = []
    for i in range(generator.size()):
        image = generator.load_image(i)
        image = preprocess_image(image)
        image, scale = generator.resize_image(image)

        # run network
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

        # correct boxes for image scale
        detections[0, :, :4] /= scale

        # change to (x, y, w, h) (MS COCO standard)
        detections[:, :, 2] -= detections[:, :, 0]
        detections[:, :, 3] -= detections[:, :, 1]


        # print(generator.image_names[i])
        im_path = os.path.join(generator.data_dir, 'JPEGImages', generator.image_names[i] + generator.image_extension)
        result_img = cv2.imread(im_path)
        # compute predicted labels and scores
        max_score = 0
        max_bbox = None
        for detection in detections[0, ...]:
            positive_labels = np.where(detection[4:] > threshold)[0]

            # append detections for each positively labeled class
            for label in positive_labels:
                image_result = {
                    # 'image_id'    : generator.image_ids[i],
                    'image_id'    : generator.image_names[i],
                    # 'category_id' : generator.classes[label],
                    'score'       : float(detection[4 + label]),
                    'bbox'        : (detection[:4]).tolist(),
                }
                if image_result["score"] > max_score:
                    max_score = image_result["score"]
                    max_bbox = image_result["bbox"]
                # append detection to results
                # results.append(image_result)
        if max_score == 0:
            print(max_score)
            continue
        detection = max_bbox
        bboxs = generator.load_annotations(i)
        # print(bboxs[0])
        bboxs = bboxs[0]
        results.append([max_score, iou(bboxs, detection)])
        cv2.rectangle(result_img, (int(detection[0]), int(detection[1])), (int(detection[0] + detection[2]), int(detection[1] + detection[3])), (255, 255, 0))
        cv2.rectangle(result_img, (int(bboxs[0]), int(bboxs[1])), (int(bboxs[2]), int(bboxs[3])), (255, 0, 0))
        cv2.imwrite('/home/liuml/retinanet/snapshots/%s.jpg'%i, result_img)
        # append image to list of processed images
        # image_ids.append(generator.image_ids[i])

        # print progress
        # print('{}/{}'.format(i, len(generator.image_ids)), end='\r')
    results = np.array(results, dtype=np.float32)
    results = results[(-results[:,0]).argsort()]
    tp = 0
    fp = 0
    aps = []
    for item in results:

        if  item[1] >  0.5:
            tp = tp + 1
            aps.append(tp / (tp + fp))
        else:
            fp = fp + 1

    if not len(results):
        return
    print(np.sum(aps) / 124)
    # write output
    # json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    # json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)
    #
    # # load results in COCO evaluation tool
    # coco_true = generator.coco
    # coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))
    #
    # # run COCO evaluation
    # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    # coco_eval.params.imgIds = image_ids
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

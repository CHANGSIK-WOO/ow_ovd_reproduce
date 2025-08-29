# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import numpy as np
import argparse
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(image_path)
    anno_image = image.copy() #detection
    sem_seg_image = image.copy()

    # ----- Semantic map overlay (if available) -----
    sem_map, sem_conf = None, None
    
    if hasattr(output, 'pred_sem_seg') and output.pred_sem_seg is not None:
        # PixelData(data: Tensor[1,H,W]) : np.ndarray
        sem = output.pred_sem_seg
        sem_map = getattr(sem, 'data', sem)
        if torch.is_tensor(sem_map):
            sem_map = sem_map.detach().cpu().numpy()
        sem_map = np.squeeze(sem_map)  # (H, W)   

    if hasattr(output, 'pred_sem_conf') and output.pred_sem_conf is not None:
        sc = output.pred_sem_conf
        sem_conf = getattr(sc, 'data', sc)
        if torch.is_tensor(sem_conf): sem_conf = sem_conf.detach().cpu().numpy()
        sem_conf = np.squeeze(sem_conf)  # (H,W) float          

    if sem_map is not None:
        #nearest
        if sem_map.shape[:2] != image.shape[:2]:
            sem_map = cv2.resize(sem_map.astype(np.uint8),
                                 (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            
        # Pallete : 0=BG (No Color), 1..K= Class
        num_classes = max(int(sem_map.max()), 0)
        # texts : [[name], [name], ...] 
        class_names = [t[0] for t in texts]

        palette = np.zeros((num_classes + 1, 3), dtype=np.uint8)
        for i in range(1, num_classes + 1):
            hue = int(179 * (i % (num_classes + 1)) / max(num_classes, 1))
            color_hsv = np.uint8([[[hue, 200, 255]]])      # (1,1,3) HSV
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            palette[i] = color_bgr       

        color_mask = palette[sem_map] 
        alpha = 0.45
        
        fg = sem_map > 0
        blended = sem_seg_image.copy()
        blended[fg] = (image[fg] * (1 - alpha) + color_mask[fg] * alpha).astype(np.uint8)
        sem_seg_image = blended

        edges = cv2.Canny((sem_map > 0).astype(np.uint8) * 255, 0, 1)
        sem_seg_image[edges > 0] = (0, 0, 0)  

            
    if sem_conf is not None:
        if sem_conf.shape[:2] != image.shape[:2]:
            sem_conf = cv2.resize(sem_conf.astype(np.float32),
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_LINEAR)                
        


        # legend_items = np.unique(sem_map)
        # legend_items = legend_items[legend_items > 0]  #
        # y0, dy, pad = 10, 18, 6
        # x0 = image.shape[1] - 10
        # for idx in legend_items[:25]: 
        #     label = class_names[idx - 1] if idx - 1 < len(class_names) else f"class_{idx-1}"
        #     txt = f"{label}"
        #     (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     x1, y1 = x0 - tw - 30, y0

        #     cv2.rectangle(sem_seg_image, (x1 - 22, y1 - th - pad), (x1 - 4, y1 + pad), palette[idx].tolist(), -1)
        #     cv2.putText(sem_seg_image, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        #     y0 += dy

    # 1) detection-only
    det_img = BOUNDING_BOX_ANNOTATOR.annotate(anno_image.copy(), detections)
    det_img = LABEL_ANNOTATOR.annotate(det_img, detections, labels=labels)
    if masks is not None:
        det_img = MASK_ANNOTATOR.annotate(det_img, detections)
    cv2.imwrite(osp.join(output_dir, osp.splitext(osp.basename(image_path))[0] + "_det.png"), det_img)

    # 2) semantic-only (+ semantic score drawn on regions)
    sem_only = sem_seg_image if sem_map is not None else image.copy()

    # Semantic score labeling (using pred_sem_conf) -------------------------
    if sem_map is not None and sem_conf is not None:
        class_names = [t[0] for t in texts]
        # Reuse the palette created above
        MIN_AREA = 200  # ignore small components
        # Class ids (exclude background 0)
        for idx in np.unique(sem_map):
            if idx <= 0:
                continue
            # Binary mask of this class
            bin_mask = (sem_map == idx).astype(np.uint8)
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            # Only draw on the largest component (change to "for c in cnts" to draw on all)
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < MIN_AREA:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

            # Semantic confidence of this class region (mean is stable, change to max if needed)
            area_mask = (sem_map == idx)
            sem_score = float(sem_conf[area_mask].mean()) if area_mask.any() else 0.0

            label = class_names[idx-1] if 0 <= idx-1 < len(class_names) else f"class_{idx-1}"
            # txt = f"{label} {sem_score:.2f}"
            txt = f"{label}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x1 = max(cx - tw//2 - 6, 0); y1 = max(cy - th//2 - 6, 0)
            x2 = min(x1 + tw + 12, sem_only.shape[1]-1); y2 = min(y1 + th + 12, sem_only.shape[0]-1)
            # Class color background box + white text
            cv2.rectangle(sem_only, (x1, y1), (x2, y2), palette[idx].tolist(), -1)
            cv2.putText(sem_only, txt, (x1+6, y1+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    # ----------------------------------------------------------------------

    cv2.imwrite(
        osp.join(output_dir, osp.splitext(osp.basename(image_path))[0] + "_sem.png"),
        sem_only
    )



    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image_path)] = anno_image
        annotations_dict[osp.basename(image_path)] = detections

        #ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)
        ANNOTATIONS_DIRECTORY = "./annotations"
        os.makedirs(ANNOTATIONS_DIRECTORY, exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(classes=[t[0] for t in texts], images=images_dict,annotations=annotations_dict).as_yolo(annotations_directory_path=ANNOTATIONS_DIRECTORY,
                                                                                                                    min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
                                                                                                                    max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
                                                                                                                    approximation_percentage=APPROXIMATION_PERCENTAGE)

    # if show:
    #     cv2.imshow('Image', image)  # Provide window name
    #     k = cv2.waitKey(0)
    #     if k == 27:
    #         # wait for ESC key to exit
    #         cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    # reparameterize texts
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for image_path in images:
        inference_detector(model,
                           image_path,
                           texts,
                           test_pipeline,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation)
        progress_bar.update()

import random
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.plots import plot_one_box
from utils.general import check_img_size, non_max_suppression, scale_coords
import cv2
from json_reader import ConfigReader
from vzone import create_zone, check_zone

config = ConfigReader()


class YOLOv7:
    def __init__(self, weights: str, image_size: int, device: str):
        self.device = device
        self.weights = weights
        self.model = attempt_load(
            self.weights, map_location=self.device)  # Model Load FP32
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, self.stride)

        if self.device != 'cpu':
            self.half = True
        else:
            self.half = False

        if self.half:
            self.model.half()  # FP16

        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        color_values = [[random.randint(0, 255) for _ in range(3)]
                        for _ in range(len(self.names))]
        self.colors = {i: color_values[i] for i in range(len(self.names))}

    def detect(self, raw_image: np.ndarray, conf_thresh=0.5, iou_thresh=0.4):
        # Run inference
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(
                self.device).type_as(next(self.model.parameters())))
        with torch.no_grad():
            image = letterbox(raw_image, self.image_size,
                              stride=self.stride)[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(self.device)
            image = image.half() if self.half else image.float()
            image /= 255.0
            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            # Inference
            detections = self.model(image, augment=False)[0]
            # Apply NMS
            detections = non_max_suppression(
                detections, conf_thresh, iou_thresh, classes=None, agnostic=False)[0]
            # Rescale boxes from img_size to raw image size
            detections[:, :4] = scale_coords(
                image.shape[2:], detections[:, :4], raw_image.shape).round()
            return detections

    def draw_bbox(self, img_raw: np.ndarray, predictions: torch.Tensor):
        try:
            for *xyxy, conf, cls in predictions:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_raw, label=label,
                             color=self.colors[int(cls)], line_thickness=1)
        except AttributeError:
            print("failed")
        return img_raw


if __name__ == '__main__':
    yolov7 = YOLOv7(weights='yolov7.pt', device='cuda', image_size=800)
    cap = cv2.VideoCapture(config.video)

    # save video
    torch.cuda.empty_cache()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))

    torch.cuda.empty_cache()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        create_zone(frame, config.zone1)

        cleared_detections = []
        zone_detections = []
        detections = yolov7.detect(frame)
        for *xyxy, conf, cls in reversed(detections):
            if yolov7.names[int(cls)] == 'person':
                cleared_detections.append([*xyxy, conf, cls])
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)
                if check_zone(center_x, center_y, config.zone1):
                    zone_detections.append([*xyxy, conf, cls])
        processed_frame = yolov7.draw_bbox(frame, cleared_detections)

        cv2.rectangle(processed_frame, (0,0), (480, 50), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"Total Person Count in Zone: {len(zone_detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        if config.save_output:
            out.write(processed_frame)

        if config.show_status:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

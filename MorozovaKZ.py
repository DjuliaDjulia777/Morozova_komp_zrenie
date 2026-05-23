""""
Программа для подсчёта автомобилей с простым трекером.
Использует YOLOv8, не требует lap/BoT-SORT.
Линия автоматически растягивается на весь кадр.
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = [self.objects[obj_id][0] for obj_id in obj_ids]

        D = np.zeros((len(obj_centroids), len(detections)), dtype=np.float32)
        for i, oc in enumerate(obj_centroids):
            for j, det in enumerate(detections):
                dc = (det[1], det[2])
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = ((detections[col][1], detections[col][2]), detections[col][0])
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(obj_centroids))) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in set(range(len(detections))) - used_cols:
            self.register(detections[col])

        return self.objects

    def register(self, det):
        bbox, cx, cy = det
        self.objects[self.next_id] = ((cx, cy), bbox)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

def line_intersection(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def parse_line(s):
    x1,y1,x2,y2 = map(int, s.split(','))
    return (x1,y1), (x2,y2)

def process_video(video_path, line_coords, direction, output_path='output_video.mp4',
                  conf=0.5, iou=0.5, show=True, save=True):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Не удалось открыть видео: {video_path}'
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video size: {w}x{h}, line coords: {line_coords}, direction: {direction}")  

    if save:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    tracker = SimpleCentroidTracker(max_disappeared=10, max_distance=50)
    counted_ids, total_crossings = set(), 0
    track_prev_centers = {}

    pt1_in, pt2_in = line_coords
    if direction == 'left_to_right':
        line_x = pt1_in[0]
        line_pt1, line_pt2 = (line_x, 0), (line_x, h)
    elif direction == 'top_to_bottom':
        line_y = pt1_in[1]
        line_pt1, line_pt2 = (0, line_y), (w, line_y)
    else:
        raise ValueError("direction must be 'left_to_right' or 'top_to_bottom'")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes
        annotated = results[0].plot() if boxes is not None and len(boxes) > 0 else frame.copy()

        if boxes is not None and len(boxes) > 0:
            car_mask = boxes.cls == 2
            car_boxes = boxes.xyxy[car_mask]
            if len(car_boxes) > 0:
                centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in car_boxes]
                detections = [(b.tolist(), int(c[0]), int(c[1])) for b,c in zip(car_boxes, centers)]
                objects = tracker.update(detections)

                for obj_id, (centroid, bbox) in objects.items():
                    cx, cy = centroid
                    if obj_id in track_prev_centers:
                        prev = track_prev_centers[obj_id]
                        if line_intersection(prev, centroid, line_pt1, line_pt2):
                            cross_ok = False
                            if direction == 'left_to_right' and prev[0] < line_x <= cx: cross_ok = True
                            elif direction == 'top_to_bottom' and prev[1] < line_y <= cy: cross_ok = True
                            if cross_ok and obj_id not in counted_ids:
                                total_crossings += 1
                                counted_ids.add(obj_id)
                    track_prev_centers[obj_id] = centroid
                    cv2.circle(annotated, (int(cx), int(cy)), 4, (0,255,255), -1)
                    cv2.putText(annotated, f'ID:{obj_id}', (int(bbox[0]), int(bbox[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            tracker.update([])  

        
        if direction == 'left_to_right':
            cv2.line(annotated, (pt1_in[0], 0), (pt1_in[0], h), (0,0,255), 4)
        elif direction == 'top_to_bottom':
            cv2.line(annotated, (0, pt1_in[1]), (w, pt1_in[1]), (0,0,255), 4)

        cv2.putText(annotated, f'Crossings: {total_crossings}', (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        if show:
            cv2.imshow('Car Counter', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        if save: out.write(annotated)

    cap.release()
    if save: out.release()
    cv2.destroyAllWindows()
    print(f'Обработано. Всего пересечений: {total_crossings}')
    return total_crossings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--line', default='300,0,300,720')
    parser.add_argument('--direction', default='left_to_right', choices=['left_to_right','top_to_bottom'])
    parser.add_argument('--output', default='output_video.mp4')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    process_video(args.video, parse_line(args.line), args.direction,
                  output_path=args.output,
                  show=not args.no_show, save=not args.no_save)
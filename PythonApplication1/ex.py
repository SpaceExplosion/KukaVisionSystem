import numpy as np
import cv2 as cv
from collections import namedtuple

FLANN_INDEX_LSH = 6
flann_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)

MIN_MATCH_COUNT = 10

PlanarTarget = namedtuple('PlaneTarget', 
                         'image, rect, keypoints, descrs, data')
TrackedTarget = namedtuple('TrackedTarget', 
                          'target, p0, p1, H, quad')

class PlaneTracker:
    def __init__(self):
        self.detector = cv.ORB_create(nfeatures=1000)
        self.matcher = cv.FlannBasedMatcher(flann_params, {})
        self.targets = []
    
    def add_target(self, image, rect, data=None):
        """Add new tracking target"""
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detector.detectAndCompute(image, None)
        if raw_descrs is None or not raw_points:
            print("В выделенной области нет особых точек — цель не добавлена.")
            return

        # Filter keypoints within rect
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)

        # np.uint8([]) дает пустой массив неверной формы, и matcher.add
        # падал внутри FLANN — проверяем заранее.
        if len(descs) < MIN_MATCH_COUNT:
            print(f"Точек в области {len(descs)}, нужно минимум "
                  f"{MIN_MATCH_COUNT} — цель не добавлена.")
            return

        descs = np.uint8(descs)
        self.matcher.add([descs])
        target = PlanarTarget(
            image=image, rect=rect, 
            keypoints=points, descrs=descs, data=data
        )
        self.targets.append(target)
    
    def track(self, frame):
        """Track targets in frame"""
        frame_points, frame_descrs = self.detector.detectAndCompute(frame, None)
        
        if len(frame_points) < MIN_MATCH_COUNT:
            return []
        
        # Match features
        matches = self.matcher.knnMatch(frame_descrs, k=2)
        matches = [m[0] for m in matches 
                  if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        
        if len(matches) < MIN_MATCH_COUNT:
            return []
        
        # Group matches by target
        matches_by_id = [[] for _ in range(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            
            target = self.targets[imgIdx]
            # knnMatch(frame_descrs): queryIdx — точки кадра, trainIdx — цели.
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            
            # Find homography
            H, status = cv.findHomography(p0, p1, cv.RANSAC, 3.0)
            # При вырожденном наборе точек OpenCV возвращает None —
            # обращение к status.ravel() падало с AttributeError.
            if H is None or status is None:
                continue
            status = status.ravel() != 0

            if status.sum() < MIN_MATCH_COUNT:
                continue
            
            p0, p1 = p0[status], p1[status]
            
            # Transform target rectangle
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            
            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        
        tracked.sort(key=lambda t: len(t.p0), reverse=True)
        return tracked

def main():
    """Демонстрация трекинга: выделите мышью область на кадре."""
    import video
    from common import RectSelector

    cap = video.create_capture(0)
    if cap is None or not cap.isOpened():
        print("Не удалось открыть источник кадров.")
        return 1

    tracker = PlaneTracker()
    # Кадр для добавления цели читается в момент выделения области:
    # RectSelector передает в callback только rect, а add_target
    # ожидает (image, rect) — без этой обертки был TypeError.
    latest = {"frame": None}

    def on_rect(rect):
        frame = latest["frame"]
        if frame is None:
            return
        tracker.add_target(frame.copy(), rect)

    # Окно должно существовать до setMouseCallback внутри RectSelector.
    cv.namedWindow('plane')
    rect_sel = RectSelector('plane', on_rect)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        latest["frame"] = frame

        vis = frame.copy()
        tracked = tracker.track(frame)

        for tr in tracked:
            cv.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
            for (x, y) in np.int32(tr.p1):
                cv.circle(vis, (int(x), int(y)), 2, (255, 255, 255))

        rect_sel.draw(vis)
        cv.imshow('plane', vis)
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
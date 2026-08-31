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
        
        # Filter keypoints within rect
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        
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
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            
            # Find homography
            H, status = cv.findHomography(p0, p1, cv.RANSAC, 3.0)
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

# Example usage
import video
from common import RectSelector

cap = video.create_capture(0)
tracker = PlaneTracker()
rect_sel = RectSelector('plane', tracker.add_target)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    vis = frame.copy()
    tracked = tracker.track(frame)
    
    for tr in tracked:
        cv.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
        for (x, y) in np.int32(tr.p1):
            cv.circle(vis, (x, y), 2, (255, 255, 255))
    
    cv.imshow('plane', vis)
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
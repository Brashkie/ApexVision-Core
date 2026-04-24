# cython: language_level=3, boundscheck=False, wraparound=False
# ApexVision-Core — Non-Max Suppression (Cython)
import numpy as np
cimport numpy as np

def nms(boxes, float iou_threshold=0.45):
    """Fast CPU NMS"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1,y1,x2,y2,scores = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0.0,xx2-xx1); h=np.maximum(0.0,yy2-yy1)
        iou=(w*h)/(areas[i]+areas[order[1:]]-w*h)
        order=order[np.where(iou<=iou_threshold)[0]+1]
    return keep

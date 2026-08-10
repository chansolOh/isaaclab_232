import numpy as np


def align_bbox(bbox):
    y_min = np.min(bbox[:,1],axis=0)
    y_min_idx = np.argwhere(bbox[:,1]==y_min)
    if y_min_idx.shape[0]>=2:
        return np.roll(bbox,-max(y_min_idx),axis=0)
    else:
        return np.roll(bbox,-y_min_idx[0],axis=0)

def lin_func(bbox,points):# bbox=2x2, points = 2xN
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    if x2-x1 == 0:
        return points[0] - x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1    
    return a*points[0] - points[1] + b 

def select_point_in_bbox(bbox,points): # bbox = 4x2, points = 2xN (x,y)
    aligned_bbox = align_bbox(bbox)
    p1 = np.where(lin_func(aligned_bbox[[3,2]],points)>=0)
    p2 = np.where(lin_func(aligned_bbox[[2,1]],points)>=0)
    p3 = np.where(lin_func(aligned_bbox[[1,0]],points)<=0)
    p4 = np.where(lin_func(aligned_bbox[[0,3]],points)<=0)

    filtered_idx = np.intersect1d(np.intersect1d(p1,p2),np.intersect1d(p3,p4))
    return filtered_idx

def select_points(bboxes,points) : # bboxes = bboxes x 2 x 4, points = 2 x N

    idx = []
    for bbox in bboxes:
        idx.append( np.array(points).T[select_point_in_bbox(bbox.T, np.array(points)[::-1] )] )


    return idx #boxes  x 2 x N
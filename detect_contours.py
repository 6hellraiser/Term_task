import numpy as np
import math
import cv2
import cv2.cv as cv
import video
import math

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
1 - toggle HSV flow visualization
 2 - toggle glitch

'''
glob_list = []
c = []

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def draw_flow(img, flow, step=16):
    global glob_list
    global c
    
    h, w = img.shape[:2]
    mat = np.zeros((h/step,w/step), np.uint8)
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)   
    vis = cv2.cvtColor(img, cv.CV_GRAY2BGR)    
    #cv2.polylines(vis, lines, 0, (0, 255, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX    
    for (x1, y1), (x2, y2) in lines:
        res = math.sqrt((x1-x2)**2+(y1-y2)**2)
        if res > 10:
            #cv2.circle(vis, (x1, y1), 3, (0, 255, 0), -1)
           # cv2.putText(vis,'0',(x1,y1), font, 0.3,(0,255,0),2,-1)
            mat[y1/step,x1/step] = 255
        #else:
            #cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
            #cv2.putText(vis,'1',(x1,y1), font, 0.3,(0,255,0),2,-1)

    one_more_list = []
    glob_list = c
    c,h = cv2.findContours(mat.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    if len(glob_list) > 0 and len(c) > 0:       
        ar =  multidim_intersect(glob_list[0],c[0])        
        if len(ar) > 0:            
            one_more_list.append(c[0])
            c1 = [cv2.approxPolyDP(cnt*step, 3, True) for cnt in c]
            cv2.drawContours(vis, c1, (-1, 3)[1 <= 0], (0,255,0), 3, cv2.CV_AA, h, abs(1))
     
    #c1 = [cv2.approxPolyDP(cnt*step, 3, True) for cnt in c]
    #cv2.drawContours(vis, c1, (-1, 3)[1 <= 0], (0,255,0), 3, cv2.CV_AA, h, abs(1))
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/math.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv.CV_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print help_message

    cam = video.create_capture(0)    
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv.CV_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    
    u = 0
   # while True:
    while u < 5:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
       
        cv2.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print 'HSV flow visualization is', ['off', 'on'][show_hsv]
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print 'glitch is', ['off', 'on'][show_glitch]
        u+=1

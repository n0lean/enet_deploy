import cv2
import moviepy
from moviepy.editor import VideoFileClip
import os
import time
import numpy as np

# camera input

'''
lock_path = 'input/lock1.txt'
os.system('rm '+ lock_path)
video_cap = cv2.VideoCapture(0)
fps = 30 #int(video_cap.get(cv2.CAP_PROP_FPS))
print fps

count = 0
while True:
    ret, frame = video_cap.read()
    if count == 0:
        frame = frame[0:424, :, :]
        frame = cv2.resize(frame, (640, 320))
        print frame.shape
        cv2.imwrite('input/cfl.jpg', frame)
        count = 10
    else:
        count -= 1
'''

# video input
clip1 = VideoFileClip('../../../CFL/IMG_4817.MOV')
clip2 = VideoFileClip('../../../CFL/IMG_4818.MOV')
clip3 = VideoFileClip('../../../CFL/IMG_4819.MOV')
clip4 = VideoFileClip('../../../CFL/IMG_4820.MOV')
lock_path = 'input/lock1.txt'
os.system('rm '+ lock_path)

count = 0
for clip in [clip1, clip2, clip3, clip4]:
    for frame in clip.iter_frames():
        if count == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[120:1080, :, :]
            frame = cv2.resize(frame, (640, 320))
            print frame.shape
            if not os.path.exists(lock_path):
                open(lock_path, 'w').close()
                print 'generator created the lock'
                assert(os.path.exists(lock_path) == True)
                print "writing to cfl.jpg"
                timer = time.time()
                cv2.imwrite('input/cfl.jpg', frame)
                print "image writing time: ", time.time()-timer
                os.system('rm '+ lock_path)
                print 'generator deleted the lock'
            count = 1
        else:
            count -= 1

'''

im_paths = os.listdir('input/Harish')
count, counter = 0, 0
for i, im in enumerate(im_paths):
    if count == 0:
        img = cv2.imread('input/Harish/Harish'+str(i+1)+'.jpg')
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        img = img[0:960, :, :]
        img = cv2.resize(img, (640, 320))
        print img.shape
        if not os.path.exists('input/lock1.txt'):
            open('input/lock1.txt', 'w').close()
            cv2.imwrite('input/test/'+str(counter)+'.jpg', img)
            counter += 1
            os.system('rm input/lock1.txt')
     
        count = 3
    else:
        count -= 1

'''

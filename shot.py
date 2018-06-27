import cv2
import sys

def capture(img_path):
    cam = cv2.VideoCapture(1)
    cam.set(3, 1280)
    cam.set(4, 720)
    ret, img = cam.read()
    if ret==False:
        print('camera error')
        assert(False)
    cv2.imwrite(img_path, img)

if __name__=='__main__':
    capture(sys.argv[1])

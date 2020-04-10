import darknet as dn
import cv2
import os, sys


def YOLO():
    net = dn.load_net(b"cfg/yolov3-obj.cfg", b"cfg/yolov3-obj_last.weights", 0)
    meta = dn.load_meta(b"data/obj.data")
    path = 'images/'
    cnt = 0
    for filename in os.listdir(path):
        print(filename)
        img = cv2.imread(path+filename)
        #print(type(img))
        ori_img = img
        img = dn.nparray_to_image(img)
        boxes = dn.detect(net, meta, img)
        #print(boxes)
        # cnt = 0
        for obj in boxes:
            classes = str(obj[0])[2:-1]
            confidence = obj[1]
            xmin = round(obj[2][0] - obj[2][2]/2)
            ymin = round(obj[2][1] - obj[2][3]/2)
            xmax = round(obj[2][0] + obj[2][2]/2)
            ymax = round(obj[2][1] + obj[2][3]/2)
            w = round(xmax - xmin)
            h = round(ymax - ymin)


            crop = ori_img[ymin:ymax, xmin:xmax]
            number = format(cnt,'05d')
            cv2.imwrite(str(number)+".jpg", crop)
            cnt += 1


if __name__ == "__main__":
    YOLO()

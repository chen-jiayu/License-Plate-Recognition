import cv2
import os,sys


path1 = 'parkingtxt/'
path2 = 'trainImg/'

count = 1
line = 1
if not os.path.exists('dataset/lexicon.txt'):
    os.mknod('dataset/lexicon.txt') 
if not os.path.exists('dataset/annotation_train.txt'):
    os.mknod('dataset/annotation_train.txt')
fp = open('dataset/lexicon.txt', 'w')
an = open('dataset/annotation_train.txt', 'w')

for filename in os.listdir(path1):
    print(count)
    f = open(path1+filename)
    content = f.readline()
    name = filename.split(".")[0]
    for filename1 in os.listdir(path2):
        name1 = filename1.split(".")[0]
        if name1 == name:
            img = cv2.imread(path2+filename1)
            cv2.imwrite('dataset/renameImg1/'+name+'_'+content+'_'+str(count)+'.jpg',img)
            fp.write(content+'\n')
            an.write('./renameImg1/'+name+'_'+content+'_'+str(count)+'.jpg'+' '+str(count)+'\n')
            count = count+1

fp.close()


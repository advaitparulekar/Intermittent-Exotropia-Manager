import cv2
import numpy as np
from Face import *

cap = cv2.VideoCapture(0)
my_face = Face(cap);

data_file = file("train.csv", 'a')
count = 0
while(cv2.waitKey(30)&0xff != 27):
    if my_face.reset():
        count +=1
        print(count)
        np.savetxt(data_file, my_face.get_features(1), fmt='%1.3f')

cap.release()
cv2.destroyAllWindows()

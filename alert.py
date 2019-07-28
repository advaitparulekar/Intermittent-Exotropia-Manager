from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from Face import *

cap = cv2.VideoCapture(0)
my_face = Face(cap);


data = np.loadtxt("train.csv", delimiter = " ")


forest = RandomForestClassifier()
X = data[:, 1:15]
Y = data[:, 0]
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
forest.fit(trainX, trainY)
print("Accuracy: \n", forest.score(testX, testY))

while(cv2.waitKey(30)&0xff != 27):
    if my_face.reset():
        if forest.predict(my_face.get_features(-1)[:, 1:15]) == 0:
            print("nvm")
        else:
            print("ALLLEEERRTTTT")

cap.release()
cv2.destroyAllWindows()

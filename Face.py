import cv2
import numpy as np
import math
from scipy import signal


np.set_printoptions(threshold='nan')

ALPHA = 2
BETA = 1

class Eye:

    def __init__(self):
        self.x = -1
        self.y = -1
        self.w = -1
        self.h = -1
        self.iris_center_x = -1
        self.iris_center_y = -1
        self.iris_radius = 1
        self.iris_radius_estimate = 0
        self.saved = False

    def features(self):
        return([self.iris_center_x, self.iris_center_y, self.iris_radius])

    def get_off_center(self, image):
         lower = np.array([0,0,0])
         upper = np.array([255,255,50])
         mask = cv2.inRange(image, lower, upper)
         cv2.imshow("eye whites", mask)

    def iris_factor(self):
        if self.w != -1:
            return self.iris_center_x/self.w
        return -1

    def test(self, image):
        for i in range(self.h/2,self.h):
            for j in range(self.w):
                image[i, j] = 0
        cv2.imshow("test",image)

    def smoothen_find(self, image):
        N = self.iris_radius_estimate*2
        #data = signal.savgol_filter(image[self.w/2], self.iris_radius_estimate, 3)
        data = np.convolve(image[self.h/2], np.ones((N,))/N, mode='valid')
        self.iris_center_y = np.argmin(data)

        #data = signal.savgol_filter(image[:, self.h/2], self.iris_radius_estimate, 3)
        data = np.convolve(image[:, self.w/2], np.ones((N,))/N, mode='valid')
        self.iris_center_x = np.argmin(data)

    def convolve_find(self, image):
        #kernel = np.ones((10,10),np.uint8)
        #image = cv2.erode(image,kernel,iterations = 1)
        min_radius = int(self.iris_radius_estimate*0.8)
        max_radius = int(self.iris_radius_estimate*1.2)

        Ox = np.zeros((2*max_radius+1, 2*max_radius+1))
        Oy = np.zeros((2*max_radius+1, 2*max_radius+1))
        W = np.zeros((2*max_radius+1, 2*max_radius+1))
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        for i in range(2*max_radius+1):
            for j in range(2*max_radius+1):
                x = i-max_radius
                y = j-max_radius
                d = x**2+y**2
                W[i, j] = 1/(math.sqrt(d)+1)
                if d > min_radius**2 and d < max_radius**2:
                    Ox[i, j] = x/float(d)
                    Oy[i, j] = -y/float(d)

        A1 = cv2.filter2D(image, -1, Sx)
        A2 = cv2.filter2D(A1, -1, Ox)
        #cv2.imshow("x derivative", A1)
        #cv2.imshow("x correlation", A2)

        B1 = cv2.filter2D(image, -1, Sy)
        B2 = cv2.filter2D(B1, -1, Oy)

        #cv2.imshow("y derivative", B1)
        #cv2.imshow("y correlation", B2)

        k = np.ones((10,10),np.uint8)
        C1 = cv2.erode(image,k,iterations = 2)
        C2 = cv2.blur(cv2.bitwise_not(C1)/3, (10, 10))
        #cv2.imshow("blur", C2)
        #print(A2+B2)

        D1 = A2/3+B2/3+C2/3
        #print("A2")
        #print(A2)
        #print("B2")
        #print(B2)
        #print("C")
        #print(C)
        D = cv2.convertScaleAbs(D1)
        max_index = np.argmax(D)
        self.iris_center_y = math.floor(max_index/eye_image.shape[0])
        self.iris_center_x = max_index - self.iris_center_y*eye_image.shape[0]

    def save_image(self, image):
        if self.saved == False:
            cv2.imwrite("eye_image.png", image)
            self.saved = True

    def process(self, image):
        image = self.convolve_find(image)
        cv2.imshow("processed image", image)
        return(image)

    def reset_iris(self, eye_image, iris_rad_est):
        self.iris_radius_estimate = 13
        eye_image = self.smoothen_find(eye_image)
        #cv2.imshow("eye edges", eye_image)
        #iris = cv2.HoughCircles(eye_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=30);
        #if iris is not None:
    #        if iris[0][0][0] > self.w/10 and iris[0][0][0] < 9*self.w/10:
    #                (self.iris_center_x, self.iris_center_y, self.radius) = iris[0][0]
    #            return True

        return True


class Face:

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def __init__(self, cap):
        self.left_eye = Eye()
        self.right_eye = Eye()
        self.x = -1
        self.y = -1
        self.w = -1
        self.h = -1
        self.cap = cap
        self.image = None

    def reset_eyes(self, eye_image):
        gray_eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        eyes = Face.eye_cascade.detectMultiScale(gray_eye_image, 1.3, 5)
        if len(eyes) != 2:
            return False
        if eyes[0][0] < eyes[1][0]:
            (self.left_eye.x, self.left_eye.y, self.left_eye.w, self.left_eye.h) = eyes[1]
            (self.right_eye.x, self.right_eye.y, self.right_eye.w, self.right_eye.h)  = eyes[0]
        else:
            (self.left_eye.x, self.left_eye.y, self.left_eye.w, self.left_eye.h) = eyes[0]
            (self.right_eye.x, self.right_eye.y, self.right_eye.w, self.right_eye.h)  = eyes[1]

        return self.left_eye.reset_iris(
            gray_eye_image[self.left_eye.y:self.left_eye.y+self.left_eye.h, \
                            self.left_eye.x:self.left_eye.x+self.left_eye.w], self.h*0.025)\
            and self.right_eye.reset_iris(
            gray_eye_image[self.right_eye.y:self.right_eye.y+self.right_eye.h,\
                            self.right_eye.x:self.right_eye.x+self.right_eye.w], self.h*0.025)

    def reset(self):
        self.ret, self.image = self.cap.read()
        gray_scale_full = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = Face.face_cascade.detectMultiScale(gray_scale_full, 1.3, 5)
        if len(faces) == 0:
            return False
        (self.x, self.y, self.w, self.h) = faces[0]
        self.reset_eyes(self.image[self.y:self.y+self.h, self.x:self.x+self.w])
        self.draw()
        return True

    def draw(self):
        cv2.rectangle(self.image, (self.x, self.y), (self.x+self.w, self.y+self.h), (255, 0, 0), 2)
        cv2.rectangle(self.image, (self.x+self.right_eye.x, self.y+self.right_eye.y), \
            (self.x+self.right_eye.x+self.right_eye.w, \
            self.y+self.right_eye.y+self.right_eye.h), \
            (255, 0, 0), 2)
        cv2.rectangle(self.image, (self.x+self.left_eye.x, self.y+self.left_eye.y), \
            (self.x+self.left_eye.x+self.left_eye.w, \
            self.y+self.left_eye.y+self.left_eye.h), \
            (255, 0, 0), 2)
        cv2.circle(self.image,(int(self.x+self.left_eye.x+self.left_eye.iris_center_x),int(self.y+self.left_eye.y+self.left_eye.iris_center_y)),2,(0,0,255),3)
        cv2.circle(self.image,(int(self.x+self.right_eye.x+self.right_eye.iris_center_x),int(self.y+self.right_eye.y+self.right_eye.iris_center_y)),2,(0,0,255),3)
        cv2.imshow("image", self.image)

    def get_features(self, label):
        return np.array([[label, self.left_eye.x,self.left_eye.y,self.left_eye.w,self.left_eye.h] \
                        +self.left_eye.features()+
                        [self.right_eye.x,self.right_eye.y,self.right_eye.w,self.right_eye.h]\
                        +self.right_eye.features()])

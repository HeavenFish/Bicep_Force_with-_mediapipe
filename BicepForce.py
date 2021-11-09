import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import math

class poseDetection():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
        self.mode, self.upBody, self.smooth, self.detectionConf, self.trackConf)
        self.weight = weight
        self.ArmWeight = ArmWeight

    # find pose

    def findPose(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 150), cv2.FILLED)
        return self.lmList

    def FindAndDraw(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        angle2 = math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y2 - y2, x2 + 50 - x2))

        angle3 = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y2 + 50 - y2, x2  - x2))

        angle2 = abs(angle2)
        angle3 = abs(angle3)

        if(88 <= abs(angle2) <= 92):
            angle2 = 90

        if (85 <= abs(angle3) <= 93):
            angle3 = 90

        if abs(angle2) != 90 or abs(angle3) != 90:
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
            cv.putText(img, str(int(angle2)), (x2 - 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            # Draw
            distant = ((x3-x2)**2+(y3-y2)**2)**0.5
            if draw:
                dis1 = int(x2 - distant * 0.1)
                dis2 = int(x1 - distant * 0.1)

                bicep_force = int((int(self.ArmWeight) * (x2 + (int((x3 - x2) / 2))) + int(self.weight) * (x3 + int(x3 * 0.2))) / dis2)
                joint_force =  int(bicep_force - int(self.weight) - int(self.ArmWeight))

                cv.line(img, (x3 + int(x3 * 0.2), y3), (x2, y2), (255, 255, 255), 3)
                cv.line(img, (dis1, y2), (x2, y2), (255, 255, 255), 3)

                # bicep force
                cv.arrowedLine(img, (x2, y2), (x1, y1), (255, 0, 0), 3)
                cv.putText(img, str(bicep_force) + "N", (x1 + 25 , y1 - 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)


                # joint force
                cv.arrowedLine(img, (dis2, y1), (dis1, y2), (0, 255, 0), 3)
                cv.putText(img, str(joint_force) + "N", (dis2-25 , y1-25), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)


                # weight force
                cv.arrowedLine(img, (x3 + int(x3 * 0.2), y1), (x3 + int(x3 * 0.2), y3), (0, 255, 0), 3)
                cv.putText(img, str(self.weight) + "N", (x3 + int(x3 * 0.2) - 50, y1 - 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                # armweight force
                cv.arrowedLine(img, (x2 + (int((x3 - x2) / 2)), y2), (x2 + int((x3 - x2) / 2), y2 + int((x3 - x2) / 2)), (0, 255, 0), 3)
                cv.putText(img, str(self.ArmWeight) + "N", (x2 + int((x3 - x2) / 2), y2 + int((x3 - x2) / 2) + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                # joint
                #cv.circle(img, (x2, y2), 5, (255, 0, 0), cv.FILLED)
                #cv.circle(img, (x2, y2), 5, (255, 0, 0), 2)

                #cv.circle(img, (int((x2 + x3) / 2), y2), 5, (255, 0, 0), cv.FILLED)
                #cv.circle(img, (int((x2 + x3) / 2), y2), 5, (255, 0, 0), 2)

                #cv.circle(img, (x1, y1), 5, (255, 0, 0), cv.FILLED)
                #cv.circle(img, (x1, y1), 5, (255, 0, 0), 2)

                #cv.circle(img, (x3 + int(x3 * 0.2), y3), 5, (255, 0, 0), cv.FILLED)
                #cv.circle(img, (x3 + int(x3 * 0.2), y3), 5, (255, 0, 0), 2)

                cv.putText(img, str(int(angle2)), (x2 - 100, y2), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

weight = int(input("Input Weight(N):"))
ArmWeight = int(input("Input ArmWeight(N):"))
detector = poseDetection()
cap = cv.VideoCapture(0)

while cap.isOpened():
    check, img = cap.read()
    img = cv.resize(img, (1080, 720))
    #cv.imshow('img', img)

    #detect arm
    img = detector.findPose(img, False)

    # detect landmark
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        # Right Arm
        angle = detector.FindAndDraw(img, 12, 14, 16)
        # Left Armq
        #angle = detector.findAngle(img, 11, 13, 15)
        #per = np.interp(angle, (210, 310), (0, 100))
        #print(angle, per)
    cv.imshow('trainer', img)

    if cv.waitKey(10) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

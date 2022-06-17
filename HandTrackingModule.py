import mediapipe as mp
import cv2
import math
import numpy as np


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5, doctorStrangeClosing=False):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHand = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        if doctorStrangeClosing:
            self.maxHands = 2

        self.hands = self.mpHand.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=0,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )

        self._open_parameters = [0] * 5
        self._closed_parameters = [0] * 5

        self._iter = 0
        self._main_distance = 35

        # 1 THUMB
        # 2 POINTER FINGER
        # 3 MIDDLE FINGER
        # 4 RING FINGER
        # 5 PINKY FINGER

        self._handNumbers = {
            1: [0, 1, 2, 3, 4],
            2: [0, 5, 6, 7, 8],
            3: [0, 9, 10, 11, 12],
            4: [0, 13, 14, 15, 16],
            5: [0, 17, 18, 19, 20]
        }

        self._check = [False, False, False]

    def get_open_calibrated(self):
        return self._open_calibrated

    def get_open_calibrated_params(self):
        return self._open_parameters

    def set_open_calibrated_params(self, params):
        self._open_parameters = params

    def get_closed_calibrated(self):
        return self._closed_calibrated

    def get_closed_calibrated_params(self):
        return self._closed_parameters

    def set_closed_calibrated_params(self, params):
        self._closed_parameters = params

    def ifCalibrated(self):
        if self._open_calibrated and self._closed_calibrated:
            return True
        return False

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)

    def getHandPoints(self, img, handNumber=0):
        lmList = []

        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNumber:
                myHand = self.results.multi_hand_landmarks[handNumber]

                for landmarkId, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmList.append([landmarkId, cx, cy])
        return lmList

    def lengthBetweenTwoPoints(self, img, first, second, showLine=True, showLineCenter=False, lmList=[]):
        if not lmList:
            lmList = self.getHandPoints(img)

        if len(lmList):
            x1, y1 = lmList[first][1], lmList[first][2]
            x2, y2 = lmList[second][1], lmList[second][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if showLine:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            if showLineCenter:
                cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length
        return 0

    def findDistance(self, img, lmList=[]):
        x = self.lengthBetweenTwoPoints(img, 0, 5, True, False, lmList)
        y = -0.0000062 * (x ** 3) + 0.0045176 * (x ** 2) - 1.1921533 * x + 135.7189041

        return y

    def getAngles(self, img, intervals):
        lmList = self.getHandPoints(img)

        angle_arr = []
        d = self.findDistance(img, lmList)

        for i in range(5):
            finger_numbers = self._handNumbers[i + 1]
            length = self.lengthBetweenTwoPoints(None, finger_numbers[0], finger_numbers[-1], False, False, lmList)

            angle_arr.append(np.interp(
                length,
                [self._main_distance*self._closed_parameters[i]/d, self._main_distance*self._open_parameters[i]/d],
                intervals[i]
            ))

        return angle_arr

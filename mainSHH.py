import cv2
from HandTrackingModule import HandDetector
import socket
import time

ip = "192.168.169.209"

open_array = [220.8144349364321, 313.91217024475304, 321.6902734197762, 305.3764276390367, 267.58657184119033]
close_array = [91.61020907050903, 136.75120479061457, 109.19050480887186, 103.4123087657179, 118.99583473857281]

def configureAngle(angles, lastAngles, delta, maxDelta):
    newAngles = []
    for i in range(5):
        if abs(angles[i] - lastAngles[i]) > maxDelta:
            # print("Ang: ", angles[i], " LA: ", lastAngles[i])
            if angles[i] > lastAngles[i]:
                newAngles.append(lastAngles[i] + maxDelta)
            else:
                newAngles.append(lastAngles[i] - maxDelta)
        elif abs(angles[i] - lastAngles[i]) < delta:
            newAngles.append(lastAngles[i])
        else:
            newAngles.append(angles[i])
    return newAngles

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    close = False

    lastAngles = [0] * 5


    detector.set_open_calibrated_params(open_array)
    detector.set_closed_calibrated_params(close_array)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, 8080))

    print("CLIENT: connected")

    while True:
        success, img = cap.read()
        detector.findHands(img)

        angles = detector.getAngles(img, [[900, 2000], [2100, 1000], [2000, 900], [2100, 1050], [1600, 500]])
        lastAngles = configureAngle(angles, lastAngles, delta=10, maxDelta=100)
        fingers = " ".join(map(lambda angle: str(int(angle)), angles))
        print(fingers)
        client.send(fingers.encode())

        cv2.imshow('Mediapipe', img)

        if cv2.waitKey(1) & 0xFF == 27 or close:
            break


if __name__ == "__main__":
    main()

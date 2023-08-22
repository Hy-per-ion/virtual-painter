import cv2
import os
import numpy as np
import HT_module as htm

fPath = "header"
myList = os.listdir(fPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{fPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (255, 0, 0)
brushThickness = 15
eraserThickness = 80

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # index and middle fingertip
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 122:
                if 150 < x1 < 350:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)  # blue
                elif 450 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)  # orange
                elif 750 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (4, 143, 11)  # green
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # black
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        else:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        img[0:122, 0:1276] = header
        # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

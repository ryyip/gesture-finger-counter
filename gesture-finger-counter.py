import cv2
from mediapipe import solutions
import math
import numpy

def main():
    cap = cv2.VideoCapture(0)
    # # Set width & height (defalut 640, 480)
    # wCam, hCam = 848, 480
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    mpHands = solutions.hands
    hands = mpHands.Hands()
    mpDraw = solutions.drawing_utils
    t = 0

    while True:
        success, img = cap.read()
        # Get camera width & height
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        if t == 0:
            print(imgWidth,imgHeight)
        t += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        fingerStr = ''
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                # for i, lm in enumerate(handLms.landmark):
                #     if i == 8:
                #         print(i, int(lm.x*imgWidth), int(lm.y*imgHeight))
            listLms = []
            for i in range(21):
                posX = hand.landmark[i].x*imgWidth
                posY = hand.landmark[i].y*imgHeight
                listLms.append([int(posX),int(posY)])
            # construct convex hull / 构建凸包
            listLms = numpy.array(listLms,dtype=numpy.int32)
            hullIndex = [0,1,2,3,6,10,14,19,18,17,10]
            hull = cv2.convexHull(listLms[hullIndex,:])
            # plot convex hull / 绘制凸包
            cv2.polylines(img,[hull],True,(0,255,0),2)
            
            # Get all the fingertip outside convex hull / 查找外部点
            tipIndex = [4,8,12,16,20]
            outFingerIndex = []
            
            for i in tipIndex:
                # Get the fingertip co-ordinate / 获取指尖坐标
                pt = (int(listLms[i][0]), int(listLms[i][1]))
                # If outside the convex hull / 是否在凸包外面
                dist = cv2.pointPolygonTest(hull,pt,True)
                if dist < 0:
                    outFingerIndex.append(i)
            
            fingerStr = getStrByOutIndex(outFingerIndex, listLms)

        cv2.putText(img, str(fingerStr), (25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        
        # Input q/Esc exit
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break


def getStrByOutIndex(outFingerIndex, listLms):
    length = len(outFingerIndex)
    res = ''
    
    if length == 1:
        if outFingerIndex[0] == 8:
            angel = getAngle(listLms[6], listLms[7], listLms[8])
            if angel < 160: 
                res = 9
            else:
                res = 1
        elif outFingerIndex[0] == 4:
            res = 'NICE'
        elif outFingerIndex[0] == 12:
            res = 'FXXK'
        elif outFingerIndex[0] == 20:
            res = 'Bad'
    elif length == 2:
        if outFingerIndex[0] == 8 and outFingerIndex[1] == 12:
            res = 2
        elif outFingerIndex[0] == 4 and outFingerIndex[1] == 20:
            res = 6
        elif outFingerIndex[0] == 4 and outFingerIndex[1] == 8:
            res = 8
    elif length == 3:
        if outFingerIndex[0] == 8 and outFingerIndex[1] == 12 and outFingerIndex[2] == 16:
            res = 3
        elif outFingerIndex[0] == 4 and outFingerIndex[1] == 8 and outFingerIndex[2] == 12:
            res = 7
        elif outFingerIndex[0] == 4 and outFingerIndex[1] == 8 and outFingerIndex[2] == 20:
            res = 'Rock & Roll'
        elif outFingerIndex[0] == 12 and outFingerIndex[1] == 16 and outFingerIndex[2] == 20:
            res = 'OK'
    elif length == 4:
        res = 4
    elif length == 5:
        res = 5
    elif length == 0:
        res = 10
    else:
        res = 0
        
    return res

# Calculate the angle betweet three point
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang)
    # return ang + 360 if ang < 0 else ang

def getAngle2(pointA, pointB, pointC):
    # vector1 from B to A 
    v1 = pointA - pointB
    # vector2 from B to C
    v2 = pointC - pointB
    # the angle between v1 and v2
    angle = numpy.dot(v1,v2)/(numpy.sqrt(numpy.sum(v1*v1))*numpy.sqrt(numpy.sum(v2*v2)))
    angle = numpy.arccos(angle)/3.14*180
    return angle

if __name__ == '__main__':
    main()

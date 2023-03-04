import cv2
import os
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = int(mode)
        self.maxHands = int(maxHands)
        self.detectionCon = int(detectionCon)
        self.trackCon = int(trackCon)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

cap = cv2.VideoCapture(0)

detector = handDetector()

fingerid= [4,8,12,16,20]        # Các vị trí ở đầu ngón tay

win_name = 'Semi-Automatic Light On Off'
imBackgroup = cv2.imread('TaiNguyen/Nen.png')

folder_path = 'TaiNguyen/Phong'
list_file = os.listdir(folder_path)
list_image = []
for i in list_file:
    image = cv2.imread(f"{folder_path}/{i}")
    list_image.append(image)

while True:
    has_frame, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    list_location = detector.findPosition(img, draw=False)     # Danh sách vị trí các điểm trên ngón tay

    if len(list_location) !=0:
        fingers= []     # 0 là đóng ngón, 1 là mở ngón
        if list_location[fingerid[0]][1] < list_location[fingerid[0] - 1][1]:     # So sánh ngón cái, nếu vị trí số 4 của trục x bé hơn vị trí thứ 3 của trục x thì ngón cái đóng
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if list_location[fingerid[id]][2] < list_location[fingerid[id] - 2][2]:       # So sánh các ngón còn lại, tương tự nhưng so sánh ở trục y
                fingers.append(1)
            else:
                fingers.append(0)
        number_of_fingers =fingers.count(1)         # Đếm xem có bao nhiêu ngón tay

        if number_of_fingers == 0:
            h, w, c = list_image[0].shape
            imBackgroup[0:h, 0:w] = list_image[0]
        elif number_of_fingers == 1:
            h, w, c = list_image[2].shape
            imBackgroup[80:80+h, 752:752+w] = list_image[2]
        elif number_of_fingers == 2:
            h, w, c = list_image[3].shape
            imBackgroup[80:80+h, 752:752+w] = list_image[3]
        elif number_of_fingers == 3:
            h, w, c = list_image[4].shape
            imBackgroup[80:80+h, 752:752+w] = list_image[4]
        elif number_of_fingers == 4:
            h, w, c = list_image[5].shape
            imBackgroup[80:80+h, 752:752+w] = list_image[5]
        elif number_of_fingers == 5:
            h, w, c = list_image[6].shape
            imBackgroup[80:80+h, 752:752+w] = list_image[6]
            
        cv2.rectangle(imBackgroup ,(53,220), (123,300), (180, 110, 66), -1)
        cv2.putText(imBackgroup ,str(number_of_fingers), (63,290), cv2.FONT_HERSHEY_PLAIN, 5, (236, 236, 236), 3)

    imBackgroup[320:320+480, 53:53+640] = img

    cv2.imshow(win_name, imBackgroup)

    if cv2.waitKey(1) == 27: # độ trễ 1/1000s , bấm Esc sẽ thoát windows
        break

cap.release()
cv2.destroyAllWindows()
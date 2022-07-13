# pip3 install opencv-python mediapipe pyautogui
import cv2
import time
import cProfile
import pyautogui
from mediapipe import solutions

# both are 0.1 second by default
pyautogui.MINIMUM_DURATION = 0.01
pyautogui.PAUSE = 0.0
INTERPOLATIONS = 10

width, height = pyautogui.size()

hand_detector = solutions.hands.Hands()
draw = solutions.drawing_utils
cap = cv2.VideoCapture(0)

# function in progress
def returnInterpolations(a, b):
    arr = []
    for i in range(1, INTERPOLATIONS):
        arr.append()

# draws the hand landmarks in the frame
def drawHand(frame, hands):
    if hands:
        for hand in hands:
            draw.draw_landmarks(frame, hand)

# temporary "noise remover"
def process(x):
	return (x - (x%10))
	# return x
	
def recognizeGesture(hands):
    if hands:
        landmark = hands[0].landmark[0]
        x = int(width * 1.25 * landmark.x) - 100
        y = int(height * 1.25 * landmark.y) - 100
        x = process(x)
        y = process(y)
        pyautogui.moveTo(x, y, 0.01, pyautogui.easeInOutQuad)
        

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    recognizeGesture(hands)
    drawHand(frame, hands)
                
    cv2.imshow("Virtual Mouse", frame)
    cv2.waitKey(1)
    

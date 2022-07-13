# pip3 install opencv-python mediapipe pyautogui
import cv2
import time
import pyautogui
from mediapipe import solutions

# both are 0.1 second by default
pyautogui.MINIMUM_DURATION = 0.01
pyautogui.PAUSE = 0.0
INTERPOLATIONS = 10
X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]

width, height = pyautogui.size()

hand_detector = solutions.hands.Hands()
draw = solutions.drawing_utils
cap = cv2.VideoCapture(0)

# wrapper function to measuring time
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

# draws the hand landmarks in the frame
def drawHand(frame, hands):
    if hands:
        for hand in hands:
            draw.draw_landmarks(frame, hand)

# temporary "noise remover"
def process(x, y):
    X, Y = X[1:], Y[1:]
    X.append(x)
    Y.append(y)
    # subhramit's function comes here

@timing
def recognizeGesture(hands):
    if hands:
        # the wrist of the 1st hand is tracked by landmark[0]
        landmark = hands[0].landmark[0]
        x = int(width * 1.25 * landmark.x) - 100
        y = int(height * 1.25 * landmark.y) - 100
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
    

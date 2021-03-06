# pip3 install opencv-python mediapipe pyautogui
import cv2
import time
import pyautogui
from mediapipe import solutions
import numpy as np

width, height = pyautogui.size()

# both are 0.1 second by default
pyautogui.MINIMUM_DURATION = 0.0
pyautogui.PAUSE = 0.0
INTERPOLATIONS = 15
SPEED = 2
X_OFFSET = 0.5 * width
Y_OFFSET = 0.5 * height

# unlike X and Y, T remains constant
X = np.ones(INTERPOLATIONS) * 100
Y = np.ones(INTERPOLATIONS) * 100
T = np.array(range(1, INTERPOLATIONS + 1), float)

hand_detector = solutions.hands.Hands()
draw = solutions.drawing_utils
cap = cv2.VideoCapture(0)

# wrapper function to measuring time
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(f.__name__, (time2 - time1) * 1000.0)
        )

        return ret

    return wrap


def restrict(x, y):
    if x >= width:
        x = width - 2
    if y >= height:
        y = height - 2
    if x <= 0:
        x = 2
    if y <= 0:
        y = 2
    return (x, y)


def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)

    # cross-deviation
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # regression coefficients
    m = SS_xy / SS_xx
    c = m_y - m * m_x

    # line is Y = mX + c
    return (c, m)


# draws the hand landmarks in the frame
def drawHand(frame, hands):
    if hands:
        for hand in hands:
            draw.draw_landmarks(frame, hand)


def moveCursor():
    global X, Y, T, MOVE_CURSOR
    (c_x, m_x) = estimate_coef(T, X)
    (c_y, m_y) = estimate_coef(T, Y)

    if abs(m_x) <= 2 and abs(m_y) <= 2:
        return

    x = int(m_x * T[-2] + c_x)
    y = int(m_y * T[-2] + c_y)
    (x, y) = restrict(x, y)
    pyautogui.moveTo(x, y, tween=pyautogui.easeInOutQuad)

    x = int(m_x * T[-1] + c_x)
    y = int(m_y * T[-1] + c_y)
    (x, y) = restrict(x, y)
    pyautogui.moveTo(x, y, tween=pyautogui.easeInOutQuad)


# temporary "noise remover"
def process(x, y):
    global X, Y, T
    X = np.append(X[1:], x)
    Y = np.append(Y[1:], y)
    moveCursor()
    # subhramit's function comes here


# @timing
def recognizeGesture(hands):
    if hands:
        # the wrist of the 1st hand is tracked by landmark[0]
        landmark = hands[0].landmark[0]
        x = int(width * SPEED * landmark.x) - X_OFFSET
        y = int(height * SPEED * landmark.y) - Y_OFFSET
        process(x, y)


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

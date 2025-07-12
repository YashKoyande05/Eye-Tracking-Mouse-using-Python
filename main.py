//Main Code
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from pynput import keyboard

# Screen settings
SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN = 10

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Smoothing
SMOOTHING_FACTOR = 0.2

# Click settings
LEFT_CLICK_HOLD = 1.5
RIGHT_CLICK_HOLD = 1.5
BLINK_THRESHOLD = 0.1
MIN_BLINK_HOLD = 0.3  # Minimum duration to consider a blink as intentional

# Scroll settings
SCROLL_MODE_TOGGLE_TIME = 3.0
NOSE_SCROLL_SENSITIVITY = 20
SCROLL_DELAY = 0.2

# Cursor update threshold
UPDATE_THRESHOLD = 0.02

# Typing mode
TYPING_TIMEOUT = 3.0
typing_mode = False
last_typing_time = 0

# Scroll mode
scroll_mode = False
scroll_toggle_start = None
last_scroll_time = 0
nose_y_baseline = None

# Calibration data
calibration_points = [
    (SCREEN_W * 0.1, SCREEN_H * 0.1),
    (SCREEN_W * 0.5, SCREEN_H * 0.1),
    (SCREEN_W * 0.9, SCREEN_H * 0.1),
    (SCREEN_W * 0.1, SCREEN_H * 0.5),
    (SCREEN_W * 0.5, SCREEN_H * 0.5),
    (SCREEN_W * 0.9, SCREEN_H * 0.5),
    (SCREEN_W * 0.1, SCREEN_H * 0.9),
    (SCREEN_W * 0.5, SCREEN_H * 0.9),
    (SCREEN_W * 0.9, SCREEN_H * 0.9)
]
calibration_data = []
calibrated = False

# Initialize
cam = cv2.VideoCapture(CAMERA_INDEX)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473
NOSE_INDEX = 1

prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2
last_valid_cursor = (prev_x, prev_y)

last_click_time = 0
blink_start_time = None
click_toggle = 'left'

def smooth_cursor(x, y):
    global prev_x, prev_y
    prev_x = (1 - SMOOTHING_FACTOR) * prev_x + SMOOTHING_FACTOR * x
    prev_y = (1 - SMOOTHING_FACTOR) * prev_y + SMOOTHING_FACTOR * y
    return int(prev_x), int(prev_y)

def draw_eye_indicators(frame, landmarks, iris_index):
    iris = landmarks[iris_index]
    x = int(iris.x * frame.shape[1])
    y = int(iris.y * frame.shape[0])
    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return x, y

def is_head_centered(landmarks, tolerance=0.05):
    if len(landmarks) < 2:
        return False
    nose = landmarks[NOSE_INDEX]
    return abs(nose.x - 0.5) < tolerance and abs(nose.y - 0.5) < tolerance

def calibrate():
    global calibration_data, calibrated
    print("✨ Calibration starting...")
    time.sleep(2)
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    calibration_data = []

    for i, (screen_x, screen_y) in enumerate(calibration_points):
        while True:
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            prompt = ""
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                if is_head_centered(landmarks): break
                else: prompt = "Center your head"
            else: prompt = "No face"
            img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(img, (int(screen_x), int(screen_y)), 25, (0, 255, 0), -1)
            cv2.putText(img, prompt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibration", img)
            cv2.waitKey(30)

        img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.circle(img, (int(screen_x), int(screen_y)), 25, (0, 255, 0), -1)
        cv2.imshow("Calibration", img)
        cv2.waitKey(1000)

        samples = []
        for _ in range(15):
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                l_x, l_y = draw_eye_indicators(frame, landmarks, LEFT_IRIS_INDEX)
                r_x, r_y = draw_eye_indicators(frame, landmarks, RIGHT_IRIS_INDEX)
                avg_x = (l_x + r_x) / 2
                avg_y = (l_y + r_y) / 2
                samples.append((avg_x, avg_y))
        if samples:
            avg_x = np.mean([s[0] for s in samples])
            avg_y = np.mean([s[1] for s in samples])
            calibration_data.append((avg_x, avg_y, screen_x, screen_y))
        else:
            calibration_data.append((SCREEN_W // 2, SCREEN_H // 2, screen_x, screen_y))

    cv2.destroyWindow("Calibration")
    calibrated = True
    print("✅ Calibration complete!")

def map_pupil_to_screen(pupil_x, pupil_y):
    x_coords, y_coords = [], []
    for calib_x, calib_y, screen_x, screen_y in calibration_data:
        dx = pupil_x - calib_x
        dy = pupil_y - calib_y
        x_coords.append(screen_x + dx * 25)
        y_coords.append(screen_y + dy * 25)
    mapped_x = np.mean(x_coords)
    mapped_y = np.mean(y_coords)
    return int(np.clip(mapped_x, MARGIN, SCREEN_W - MARGIN)), int(np.clip(mapped_y, MARGIN, SCREEN_H - MARGIN))

def update_status(msg):
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Status", img)
    cv2.moveWindow("Status", SCREEN_W - 320, SCREEN_H - 220)

def on_press(key):
    global typing_mode, last_typing_time
    last_typing_time = time.time()
    typing_mode = True

listener = keyboard.Listener(on_press=on_press)
listener.start()

calibrate()
print("🔥 Starting tracking...")

while True:
    ret, frame = cam.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if time.time() - last_typing_time > TYPING_TIMEOUT:
        typing_mode = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_openness = abs(landmarks[159].y - landmarks[145].y)
        right_openness = abs(landmarks[386].y - landmarks[374].y)
        nose = landmarks[NOSE_INDEX]
        nose_y = int(nose.y * frame.shape[0])

        mouth_openness = abs(landmarks[13].y - landmarks[14].y)
        if mouth_openness > 0.05:
            if scroll_toggle_start is None:
                scroll_toggle_start = time.time()
            elif time.time() - scroll_toggle_start > SCROLL_MODE_TOGGLE_TIME:
                scroll_mode = not scroll_mode
                nose_y_baseline = nose_y
                scroll_toggle_start = None
        else:
            scroll_toggle_start = None

        if scroll_mode:
            if nose_y_baseline is not None and time.time() - last_scroll_time > SCROLL_DELAY:
                if nose_y < nose_y_baseline - NOSE_SCROLL_SENSITIVITY:
                    pyautogui.scroll(20)
                    last_scroll_time = time.time()
                elif nose_y > nose_y_baseline + NOSE_SCROLL_SENSITIVITY:
                    pyautogui.scroll(-20)
                    last_scroll_time = time.time()
        else:
            if not typing_mode and left_openness > UPDATE_THRESHOLD and right_openness > UPDATE_THRESHOLD:
                l_x, l_y = draw_eye_indicators(frame, landmarks, LEFT_IRIS_INDEX)
                r_x, r_y = draw_eye_indicators(frame, landmarks, RIGHT_IRIS_INDEX)
                mx, my = map_pupil_to_screen((l_x + r_x)/2, (l_y + r_y)/2)
                smooth_x, smooth_y = smooth_cursor(mx, my)
                last_valid_cursor = (smooth_x, smooth_y)
                pyautogui.moveTo(smooth_x, smooth_y)
            else:
                pyautogui.moveTo(last_valid_cursor[0], last_valid_cursor[1])

            # Click logic: block while typing, and filter out natural blinks
            now = time.time()
            if not typing_mode:
                if left_openness < BLINK_THRESHOLD and right_openness < BLINK_THRESHOLD:
                    if blink_start_time is None:
                        blink_start_time = now
                    elif now - blink_start_time > MIN_BLINK_HOLD:
                        if click_toggle == 'left' and now - last_click_time > LEFT_CLICK_HOLD:
                            pyautogui.click(button='left')
                            update_status("Left Click Triggered!")
                            click_toggle = 'right'
                            last_click_time = now
                            blink_start_time = None
                        elif click_toggle == 'right' and now - last_click_time > RIGHT_CLICK_HOLD:
                            pyautogui.click(button='right')
                            update_status("Right Click Triggered!")
                            click_toggle = 'left'
                            last_click_time = now
                            blink_start_time = None
                else:
                    blink_start_time = None

    if typing_mode:
        update_status("Typing mode")
    elif scroll_mode:
        update_status("Scroll mode")
    else:
        update_status("Tracking")

    cv2.imshow("Pupil Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

listener.stop()
cam.release()
cv2.destroyAllWindows()

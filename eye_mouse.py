import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np


def get_blink_ratio(landmarks, indices):
    top = landmarks[indices[0]]
    bottom = landmarks[indices[1]]
    inner = landmarks[indices[2]]
    outer = landmarks[indices[3]]
    
    ver_dist = np.hypot(top.x - bottom.x, top.y - bottom.y)
    hor_dist = np.hypot(inner.x - outer.x, inner.y - outer.y)
    
    return ver_dist / (hor_dist + 1e-6)

def is_inside(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

last_click_time = 0
click_cooldown = 1.0 # seconds

cap = cv2.VideoCapture(0)

# Smoothing variables
smooth_factor = 7 
# Initialize with current mouse position to prevent startup jump
start_x, start_y = pyautogui.position()
ploc_x, ploc_y = start_x, start_y
cloc_x, cloc_y = start_x, start_y

active = True 
calibrated = False
eye_min_x, eye_max_x = 1.0, 0.0
eye_min_y, eye_max_y = 1.0, 0.0

eye_min_y, eye_max_y = 1.0, 0.0

# Button Coordinates (x1, y1, x2, y2)
btn_open = (20, 20, 120, 70)
btn_close = (140, 20, 240, 70)
btn_quit = (260, 20, 360, 70)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    # Draw Buttons
    cv2.rectangle(frame, (btn_open[0], btn_open[1]), (btn_open[2], btn_open[3]), (0, 255, 0), -1)
    cv2.putText(frame, "OPEN", (btn_open[0]+10, btn_open[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (btn_close[0], btn_close[1]), (btn_close[2], btn_close[3]), (0, 0, 255), -1)
    cv2.putText(frame, "CLOSE", (btn_close[0]+10, btn_close[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # QUIT Button (Gray)
    cv2.rectangle(frame, (btn_quit[0], btn_quit[1]), (btn_quit[2], btn_quit[3]), (100, 100, 100), -1)
    cv2.putText(frame, "QUIT", (btn_quit[0]+10, btn_quit[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        
        id = 468 
        pt = landmarks[id]
        
        cx = int(pt.x * frame_w)
        cy = int(pt.y * frame_h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
        
        blink_ratio = get_blink_ratio(landmarks, [386, 374, 362, 263])
        blink_thresh = 0.18
        is_blinking = blink_ratio < blink_thresh
        
        cv2.putText(frame, f"Blink: {blink_ratio:.2f}", (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Button Interactions
        current_time = time.time()
        if is_blinking and (current_time - last_click_time > click_cooldown):
            if is_inside(cx, cy, btn_open):
                active = True
                cv2.putText(frame, "ACTIVATED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_click_time = current_time
                
            elif is_inside(cx, cy, btn_close):
                active = False
                cv2.putText(frame, "DEACTIVATED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                last_click_time = current_time
                
            elif is_inside(cx, cy, btn_quit):
                cv2.putText(frame, "QUITTING...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Eye Controlled Mouse', frame)
                cv2.waitKey(500)
                break

        if active:
            if not calibrated:
                cv2.putText(frame, "CALIBRATION MODE", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "1. Look at corners", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "2. Press 's' to SAVE", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if pt.x < eye_min_x: eye_min_x = pt.x
                if pt.x > eye_max_x: eye_max_x = pt.x
                if pt.y < eye_min_y: eye_min_y = pt.y
                if pt.y > eye_max_y: eye_max_y = pt.y
                
                rect_x = int(eye_min_x * frame_w)
                rect_y = int(eye_min_y * frame_h)
                rect_w = int((eye_max_x - eye_min_x) * frame_w)
                rect_h = int((eye_max_y - eye_min_y) * frame_h)
                cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 0), 2)

            else:
                denom_x = (eye_max_x - eye_min_x) if (eye_max_x - eye_min_x) > 0 else 0.01
                denom_y = (eye_max_y - eye_min_y) if (eye_max_y - eye_min_y) > 0 else 0.01
                
                norm_x = (pt.x - eye_min_x) / denom_x
                norm_y = (pt.y - eye_min_y) / denom_y
                
                norm_x = np.clip(norm_x, 0, 1)
                norm_y = np.clip(norm_y, 0, 1)
                
                target_x = screen_w * norm_x
                target_y = screen_h * norm_y
                
                # Smoothing with Deadzone
                dist = np.linalg.norm(np.array([target_x, target_y]) - np.array([ploc_x, ploc_y]))
                
                if dist > 25: 
                    cloc_x = ploc_x + (target_x - ploc_x) / smooth_factor
                    cloc_y = ploc_y + (target_y - ploc_y) / smooth_factor
                    
                    pyautogui.moveTo(cloc_x, cloc_y)
                    ploc_x, ploc_y = cloc_x, cloc_y
                
                if is_blinking and (current_time - last_click_time > click_cooldown):
                    cv2.putText(frame, "CLICK!", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    pyautogui.click()
                    last_click_time = current_time
        else:
            cv2.putText(frame, "PAUSED", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow('Eye Controlled Mouse', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        calibrated = True
    elif key == ord('c'):
        calibrated = False
        eye_min_x, eye_max_x = 1.0, 0.0
        eye_min_y, eye_max_y = 1.0, 0.0

cap.release()
cv2.destroyAllWindows()

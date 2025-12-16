"""
Advanced Virtual Mouse Control using Hand Gestures
Гарын хөдөлгөөнөөр хулганыг удирдах систем - Advanced Edition
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()


pyautogui.FAILSAFE = False  # Хулганыг булан руу аваачихад програм зогсохгүй байх тохиргоо
pyautogui.PAUSE = 0 # Хулганы үйлдэл хоорондын хүлээлтийг 0 болгох (илүү хурдан) 

cam_width, cam_height = 640, 480


prev_x, prev_y = 0, 0
smoothing = 7  

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque


screen_width, screen_height = pyautogui.size()
cam_width, cam_height = 640, 480

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_angle(a, b, c):
    """Calculate angle between three points (from GitHub repo)
    Гурван цэгийн хоорондох өнцгийг тооцоолох (GitHub репо-оос авсан)
    """
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

class VirtualMouse:
    def __init__(self):
        """Initialize mouse settings and variables - Хулганы тохиргоо болон хувьсагчдыг эхлүүлэх"""
        self.hands = mp_hands.Hands(
            static_image_mode=False, # Видео горимд ажиллах (зураг биш)
            max_num_hands=1, # Зөвхөн нэг гар таних
            min_detection_confidence=0.7, # Танилтын нарийвчлал (70%)
            min_tracking_confidence=0.5 # Дагах нарийвчлал (50%)
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)
        
        
        self.prev_x = 0
        self.prev_y = 0
        self.click_cooldown = 0
        self.last_gesture = "none"
        self.last_finger_status = [0, 0, 0, 0, 0]  
        
        
        self.gesture_history = deque(maxlen=10)  
        self.fps_history = deque(maxlen=30)  
        self.confidence_scores = deque(maxlen=30)
        self.gesture_start_time = {}
        self.gesture_durations = {}
        
     
        self.total_clicks = 0
        self.total_moves = 0
        self.total_double_clicks = 0  
        self.total_drags = 0  
        self.session_start = time.time()
        
     
        self.is_dragging = False
        self.drag_start_pos = None
        
    
        self.last_finger_status = [0, 0, 0, 0, 0]
        
    
        self.show_advanced_info = True
        self.show_trails = True
        self.trail_points = deque(maxlen=20)
        
    
        self.colors = {
            'move': (0, 255, 0),           # Green - 1 finger
            'left_click': (0, 255, 255),   # Cyan - 2 fingers
            'drag_start': (255, 0, 255),   # Magenta - 3 fingers (start)
            'drag_hold': (255, 0, 255),    # Magenta - 3 fingers (holding)
            'double_click': (255, 255, 0), # Yellow - Thumb up
            'stop': (128, 128, 128),       # Gray - Open hand
            'none': (255, 255, 255)        # White
        }
        
    def get_finger_status(self, landmarks):
        """Check which fingers are extended - Аль хуруунууд тэнийсэн байгааг шалгах"""
        finger_tips = [8, 12, 16, 20] 
        finger_status = []
        
        for tip in finger_tips:
            
            # Хурууны үзүүр нь үенээсээ доор байгаа эсэхийг шалгах (Y тэнхлэг доошоо өсдөг)
            if landmarks[tip].y < landmarks[tip - 2].y:
                finger_status.append(1)  # Хуруу тэнийсэн
            else:
                finger_status.append(0)  # Хуруу нугалсан  
                
        
        if landmarks[4].x < landmarks[3].x:  
            finger_status.insert(0, 1)
        else:
            finger_status.insert(0, 0)
            
        return finger_status
    
    def detect_gesture(self, landmarks):
        """Detect specific hand gestures - SIMPLIFIED & PRACTICAL! - Гарын дохиог таних (Хялбаршуулсан & Практик)"""
        finger_status = self.get_finger_status(landmarks)
        
    
        self.last_finger_status = finger_status
        
     
        # Зөвхөн долоовор болон дунд хуруу тэнийсэн бол -> Зүүн товч дарах
        if finger_status == [0, 1, 1, 0, 0]:
            return "left_click"
        
        
        # Долоовор, дунд, ядам хуруунууд тэнийсэн бол -> Чирэх үйлдэл
        elif finger_status == [0, 1, 1, 1, 0]:
            if self.is_dragging:
                return "drag_hold" # Чирж байгаа үед
            else:
                return "drag_start" # Чирч эхлэх үед
        
      
        # Зөвхөн эрхий хуруу тэнийсэн бол -> Давхар дарах
        elif finger_status == [1, 0, 0, 0, 0]:
            return "double_click"
        
       
        # Зөвхөн долоовор хуруу тэнийсэн бол -> Курсор хөдөлгөх
        if finger_status[1] == 1:  
            return "move"
        
        return "none"

    def draw_finger_status_overlay(self, frame):
        """Draw a compact finger-status overlay (always visible). - Хурууны төлөвийг харуулах цонхыг зурах (үргэлж харагдана)"""
        if not hasattr(self, 'last_finger_status'):
            return

        
        h, w = frame.shape[:2]
        box_w, box_h = 220, 120
        x0, y0 = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

       
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for i, (name, status) in enumerate(zip(finger_names, self.last_finger_status)):
            status_text = "UP" if status == 1 else "DOWN"
            status_color = (0, 220, 0) if status == 1 else (180, 180, 180)
            y = y0 + 22 + i * 20
            cv2.putText(frame, f"{name}: ", (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, status_text, (x0 + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        
        arr_text = str(self.last_finger_status)
        cv2.putText(frame, f"Array: {arr_text}", (x0 + 8, y0 + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    def is_pinching(self, landmarks, finger1_tip, finger2_tip):
        """Check if two fingers are pinching (touching) - Хоёр хуруу чимхсэн эсэхийг шалгах (хүрэлцэх)"""
        tip1 = landmarks[finger1_tip]
        tip2 = landmarks[finger2_tip]
        
        distance = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2)
        
        return distance < 0.05  
    
    def move_cursor(self, index_finger):
        """Move cursor based on index finger position - Долоовор хурууны байрлалаар курсорыг хөдөлгөх"""
        x = int(index_finger.x * screen_width)
        y = int(index_finger.y * screen_height)
        
        
        # Курсорын хөдөлгөөнийг зөөлрүүлэх (Smoothing)
        curr_x = self.prev_x + (x - self.prev_x) / smoothing
        curr_y = self.prev_y + (y - self.prev_y) / smoothing
        
  
        pyautogui.moveTo(curr_x, curr_y, duration=0)
        
      
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        return int(curr_x), int(curr_y)
    
    def run(self):
        """Main loop with advanced features - Үндсэн ажиллагааны давталт (дэвшилтэт боломжуудтай)"""
        print("🖱️ Advanced Virtual Mouse Control Started!")
        print("📹 Camera feed opening...")
        print("\n🖐️ Gestures (IMPROVED - Илүү амархан!):")
        print("  ☝️  Index finger up = Move cursor")
        print("  ✊ Fist (all closed) = Left Click (NEW - EASY!)")
        print("  ✌️  Peace sign (2 fingers) = Right Click (NEW!)")
        print("  🖐️  Open hand (5 fingers) = Stop")
        print("  💡 NO MORE PINCHING - Way easier!")
        print("\n⌨️  Keyboard Shortcuts:")
        print("  'q' = Quit")
        print("  'i' = Toggle info display")
        print("  't' = Toggle trails")
        print("  's' = Save screenshot")
        print("  'r' = Reset statistics\n")
        
        print("🎮 NEW SIMPLE GESTURES:")
        print("  ☝️  1 finger (index) = Move cursor")
        print("  ✌️  2 fingers (index+middle) = Left Click")
        print("  🎯 3 fingers (index+middle+ring) = DRAG & DROP")
        print("  👍 Thumb up = Double Click (open files)")
        print("  🖐️  Open hand (5 fingers) = Stop")
        print("  💡 SUPER SIMPLE & PRACTICAL!\n")
        
        frame_count = 0
        
        while True:
            frame_start = time.time()
            success, frame = self.cap.read()
            if not success:
                print("❌ Failed to capture frame - Камерын дүрсийг авч чадсангүй")
                break
            
            frame_count += 1
            
           
            # Дүрсийг толь шиг эргүүлэх
            frame = cv2.flip(frame, 1)
            
            # BGR өнгөний орон зайг RGB руу хөрвүүлэх (MediaPipe-д зориулж)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Гарыг илрүүлэх
            results = self.hands.process(rgb_frame)
            
            gesture = "none"
            cursor_pos = None
            hand_confidence = 0
            
           
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                   
                    if results.multi_handedness:
                        hand_confidence = results.multi_handedness[0].classification[0].score
                        self.confidence_scores.append(hand_confidence)
                    
                    
                    self.draw_enhanced_landmarks(frame, hand_landmarks)
                    
                    
                    # Гарын цэгүүдийг авах
                    landmarks = hand_landmarks.landmark
                    
                    # Дохиог таних
                    gesture = self.detect_gesture(landmarks)
                    
                   
                    self.gesture_history.append(gesture)
                    
                  
                    if gesture != self.last_gesture:
                        if self.last_gesture in self.gesture_start_time:
                            duration = time.time() - self.gesture_start_time[self.last_gesture]
                            if self.last_gesture not in self.gesture_durations:
                                self.gesture_durations[self.last_gesture] = []
                            self.gesture_durations[self.last_gesture].append(duration)
                        self.gesture_start_time[gesture] = time.time()
                    
                    
                    if gesture in ["move"]:
                        index_finger = landmarks[8]  # Долоовор хурууны үзүүр
                        cursor_pos = self.move_cursor(index_finger)
                        self.total_moves += 1
                        
                        
                        if self.show_trails:
                            finger_x = int(index_finger.x * cam_width)
                            finger_y = int(index_finger.y * cam_height)
                            self.trail_points.append((finger_x, finger_y))
                        
                    elif gesture == "left_click" and self.click_cooldown == 0:
                        # Зүүн товч дарах үйлдэл
                        pyautogui.click()
                        self.click_cooldown = 15 # Дараагийн даралт хүртэл хүлээх хугацаа
                        self.total_clicks += 1
                        print(f"✌️ 2-Finger Click! (Total: {self.total_clicks})")
                       
                        cv2.circle(frame, (cam_width//2, cam_height//2), 60, (255, 255, 0), -1)
                    
                    elif gesture == "drag_start":
                        # Чирэх үйлдлийг эхлүүлэх
                        if not self.is_dragging:
                            pyautogui.mouseDown() # Хулганы товчийг дараад барих
                            self.is_dragging = True
                            self.total_drags += 1
                            self.drag_start_pos = pyautogui.position()
                            print(f"🎯 3-Finger DRAG START! Keep 3 fingers, move to drag!")
                        
                            cv2.circle(frame, (cam_width//2, cam_height//2), 70, (255, 0, 255), -1)
                        
                    elif gesture == "drag_hold":
                       
                        if self.is_dragging:
                            cv2.putText(frame, "DRAGGING... (3 FINGERS)", (cam_width//2 - 150, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                    
                    elif gesture == "move":
                        # Хэрэв чирж байсан бол чирэх үйлдлийг зогсоох (зөвхөн 1 хуруу үлдсэн үед)
                        if self.is_dragging:
                            pyautogui.mouseUp() # Хулганы товчийг тавих
                            self.is_dragging = False
                            drag_end = pyautogui.position()
                            if self.drag_start_pos:
                                distance = int(np.sqrt((drag_end[0]-self.drag_start_pos[0])**2 + 
                                                      (drag_end[1]-self.drag_start_pos[1])**2))
                                print(f"🎯 DRAG END! Dropped at 1-finger. Distance: {distance}px")
                            self.drag_start_pos = None
                    
                    elif gesture == "double_click" and self.click_cooldown == 0:
                        pyautogui.doubleClick() # Давхар дарах
                        self.click_cooldown = 25
                        self.total_double_clicks += 1
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                        print(f"👍 Thumb Double Click! (Total: {self.total_double_clicks})")
                        
                        cv2.circle(frame, (cam_width//2, cam_height//2), 80, (0, 255, 255), -1)
                    
                    elif gesture == "stop":
                        
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            print("🖐️ Stop - Drag cancelled!")
                    
                    self.last_gesture = gesture
            
            else:
                
                if self.is_dragging:
                    pyautogui.mouseUp() # Гар алга болсон үед чирэхийг зогсоох
                    self.is_dragging = False
                    print("👋 Hand lost - Drag released")
            
            
            if self.click_cooldown > 0:
                self.click_cooldown -= 1
            
           
            if self.show_trails and len(self.trail_points) > 1:
                for i in range(1, len(self.trail_points)):
                    alpha = i / len(self.trail_points)
                    thickness = int(2 + alpha * 3)
                    cv2.line(frame, self.trail_points[i-1], self.trail_points[i], 
                            self.colors.get(gesture, (255, 255, 255)), thickness)
            
           
            frame_time = time.time() - frame_start
            fps = 1 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            
            if self.show_advanced_info:
                self.draw_info_overlay(frame, gesture, cursor_pos, hand_confidence, avg_fps, frame_count)
            else:
               
                cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors.get(gesture, (255, 255, 255)), 2)
            
            
            self.draw_gesture_indicator(frame, gesture)

            
            self.draw_finger_status_overlay(frame)

            
            cv2.imshow('Advanced Virtual Mouse - Гарын хулгана', frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.show_advanced_info = not self.show_advanced_info
                print(f"ℹ️ Info display: {'ON' if self.show_advanced_info else 'OFF'}")
            elif key == ord('t'):
                self.show_trails = not self.show_trails
                print(f"🌟 Trails: {'ON' if self.show_trails else 'OFF'}")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                self.reset_statistics()
                print("🔄 Statistics reset!")
        
        
        self.print_session_summary()
        
     
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✅ Virtual Mouse stopped")
    
    def draw_enhanced_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks with enhanced visuals - Гарын цэгүүдийг сайжруулсан байдлаар зурах"""
       
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            
            start_point = (int(start.x * cam_width), int(start.y * cam_height))
            end_point = (int(end.x * cam_width), int(end.y * cam_height))
            
            # Холбоос шугамыг зурах
            cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
        
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * cam_width)
            y = int(landmark.y * cam_height)
            
           
            # Хурууны үзүүрүүдийг тодруулж зурах
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    def draw_info_overlay(self, frame, gesture, cursor_pos, confidence, fps, frame_count):
        """Draw comprehensive information overlay - Дэлгэрэнгүй мэдээллийн самбарыг зурах"""
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (5, 5), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 25
        line_height = 22
        
       
        color = self.colors.get(gesture, (255, 255, 255))
        cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        if hasattr(self, 'last_finger_status'):
            finger_icons = ["👍", "☝️", "✌️", "💍", "🤙"]
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
            cv2.putText(frame, "Finger Status:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_offset += line_height
            
            for i, (name, status) in enumerate(zip(finger_names, self.last_finger_status)):
                status_text = "UP" if status == 1 else "DOWN"
                status_color = (0, 255, 0) if status == 1 else (100, 100, 100)
                cv2.putText(frame, f"  {name}: {status_text}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                y_offset += 18
            
            
            cv2.putText(frame, f"Array: {self.last_finger_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
        
        
        if cursor_pos:
            cv2.putText(frame, f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]})", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height
        
       
        conf_text = f"Confidence: {confidence*100:.1f}%"
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
        cv2.putText(frame, conf_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
       
        session_time = int(time.time() - self.session_start)
        cv2.putText(frame, f"Session: {session_time}s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Clicks: {self.total_clicks}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Double: {self.total_double_clicks}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Drags: {self.total_drags}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Moves: {self.total_moves}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        if len(self.gesture_history) > 0:
            recent = list(self.gesture_history)[-5:]
            history_text = " > ".join([g[:4] for g in recent])
            cv2.putText(frame, f"History: {history_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.putText(frame, "Press 'i' to toggle info", (10, cam_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    def draw_gesture_indicator(self, frame, gesture):
        """Draw a visual indicator for current gesture - Одоогийн дохионы дүрсийг зурах"""
        radius = 40
        center_x = cam_width - radius - 20
        center_y = radius + 20
        
        color = self.colors.get(gesture, (255, 255, 255))
        
    
        pulse = int(10 * math.sin(time.time() * 5))
        current_radius = radius + pulse
        

        cv2.circle(frame, (center_x, center_y), current_radius, color, 3)
        cv2.circle(frame, (center_x, center_y), current_radius - 10, color, -1)
        
        icon_map = {
            'move': '☝️',
            'left_click': '✊',
            'double_click': '�',
            'right_click': '✌️',
            'drag': '🎯',
            'stop': '🖐️',
            'none': '?'
        }
        
        icon = icon_map.get(gesture, '?')
        cv2.putText(frame, gesture[:4].upper(), (center_x - 20, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def reset_statistics(self):
        """Reset all statistics - Бүх статистикийг дахин эхлүүлэх"""
        self.total_clicks = 0
        self.total_moves = 0
        self.total_double_clicks = 0
        self.total_drags = 0
        self.session_start = time.time()
        self.gesture_durations.clear()
    
    def print_session_summary(self):
        """Print session statistics - Сессийн статистикийг хэвлэх"""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        
        duration = int(time.time() - self.session_start)
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🖱️  Total Clicks: {self.total_clicks}")
        print(f"👍 Double Clicks: {self.total_double_clicks}")
        print(f"🎯 Total Drags: {self.total_drags}")
        print(f"↔️  Total Moves: {self.total_moves}")
        
        if self.confidence_scores:
            avg_conf = sum(self.confidence_scores) / len(self.confidence_scores)
            print(f"📈 Avg Confidence: {avg_conf*100:.1f}%")
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"🎬 Avg FPS: {avg_fps:.1f}")
        
        print("\n🖐️ Gesture Usage:")
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.gesture_history)) * 100
            print(f"  {gesture:12s}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*50)

if __name__ == "__main__":
    try:
        mouse = VirtualMouse()
        mouse.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

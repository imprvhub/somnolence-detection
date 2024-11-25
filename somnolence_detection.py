import cv2
from scipy.spatial import distance
import mediapipe as mp
import numpy as np

class DrowsinessDetector:
    def __init__(self):
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.EAR_THRESH = 0.25
        self.CLOSED_EYES_FRAME = 20
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.counter = 0
        
    def calculate_EAR(self, eye_points):
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
        
    def put_text_with_background(self, img, text, position, scale=1, color=(255,255,255), thickness=2):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        
        padding = 10
        bg_color = (0,0,255) if "Somnolence" in text else (0,0,0)
        text_color = (255,255,255) if "Somnolence" in text else color
        
        cv2.rectangle(img, 
                     (position[0] - padding, position[1] - text_height - padding),
                     (position[0] + text_width + padding, position[1] + padding),
                     bg_color, 
                     -1)
        
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness)
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])
            
            left_eye_points = mesh_points[self.LEFT_EYE]
            right_eye_points = mesh_points[self.RIGHT_EYE]
            
            left_ear = self.calculate_EAR(left_eye_points)
            right_ear = self.calculate_EAR(right_eye_points)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
            
            if avg_ear < self.EAR_THRESH:
                self.counter += 1
                if self.counter >= self.CLOSED_EYES_FRAME:
                    self.put_text_with_background(frame, "Somnolence ALERT!", (10, 50), 
                                                scale=1.2, color=(0, 0, 255), thickness=2)
            else:
                self.counter = 0
                
            self.put_text_with_background(frame, 
                                        f"EAR (Eye Aspect Ratio): {avg_ear:.2f}", 
                                        (10, 100), 
                                        scale=0.7, 
                                        color=(255, 255, 255), 
                                        thickness=2)
        
        return frame
    
    def start_detection(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Somnolence Detection', cv2.WINDOW_NORMAL)
        
        if not cap.isOpened():
            raise Exception("Could not open camera")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            cv2.imshow('Somnolence Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.start_detection()
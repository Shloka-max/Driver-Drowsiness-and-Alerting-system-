import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound3 import playsound

class DriverDrowsinessDetector:
    def __init__(self, shape_predictor_path, alert_sound_sleep, alert_sound_drowsy):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.alert_sound_sleep = alert_sound_sleep
        self.alert_sound_drowsy = alert_sound_drowsy
        self.status = ""
        self.color = (0, 0, 0)
        self.sleep = 0
        self.drowsy = 0
        self.active = 0

    def compute_distance(self, ptA, ptB):
        return np.linalg.norm(ptA - ptB)

    def blinked(self, a, b, c, d, e, f):
        up = self.compute_distance(b, d) + self.compute_distance(c, e)
        down = self.compute_distance(a, f)
        ratio = up / (2.0 * down)

        if ratio > 0.25:
            return 2
        elif 0.21 < ratio <= 0.25:
            return 1
        else:
            return 0

    def update_status(self, frame, landmarks):
        left_blink = self.blinked(landmarks[36], landmarks[37],
                                  landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = self.blinked(landmarks[42], landmarks[43],
                                   landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            self.sleep += 1
            self.drowsy = 0
            self.active = 0
            if self.sleep > 6:
                self.status = "SLEEPING !!!"
                playsound(self.alert_sound_sleep)
                self.color = (255, 0, 0)
        elif left_blink == 1 or right_blink == 1:
            self.sleep = 0
            self.drowsy += 1
            self.active = 0
            if self.drowsy > 6:
                self.status = "drowsy !!!"
                playsound(self.alert_sound_drowsy)
                self.color = (0, 0, 255)
        else:
            self.sleep = 0
            self.active += 1
            self.drowsy = 0
            if self.active > 6:
                self.status = "Active :)"
                self.color = (0, 255, 0)

        block_message = f"{self.status}"
        text_size, _ = cv2.getTextSize(block_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int((frame.shape[0] + text_size[1]) / 2)
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 10, text_y + 10), self.color, -1)
        cv2.putText(frame, block_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    def run(self, video_capture_index=0):
        cap = cv2.VideoCapture(video_capture_index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = self.predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)
                self.update_status(frame, landmarks)

                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            cv2.imshow("Result of detector", frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    shape_predictor_path = r"C:\Users\nisha\Downloads\Driver-Drowsiness-Detection-master\Driver-Drowsiness-Detection-master\shape_predictor_68_face_landmarks.dat"
    alert_sound_sleep = r'C:\Users\nisha\Downloads\Driver-Drowsiness-Detection-master\Driver-Drowsiness-Detection-master\alert-sound.mp3'
    alert_sound_drowsy = r'C:\Users\nisha\Downloads\Driver-Drowsiness-Detection-master\Driver-Drowsiness-Detection-master\take_a_break.mp3'
    detector = DriverDrowsinessDetector(shape_predictor_path, alert_sound_sleep, alert_sound_drowsy)
    detector.run()

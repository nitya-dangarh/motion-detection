import cv2
import imutils
import os
from plyer import notification
import pygame
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSlider, QListWidget, QListWidgetItem
import datetime

# Suppress macOS warning
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Initialize Pygame for audio
pygame.init()

class MotionLogger:
    def __init__(self):
        self.motion_events = []

    def add_event(self):
        timestamp = datetime.datetime.now()
        self.motion_events.append({"timestamp": timestamp})

    def export_to_file(self, filename="motion_events.log"):
        with open(filename, "w") as file:
            for event in self.motion_events:
                file.write(f"{event['timestamp']}\n")

class MotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.normal_mode = True
        self.start_frame = None
        self.alarm_mode = False
        self.roi_defined = False
        self.roi_start = None
        self.roi_end = None

        self.alarm_sound = "mixkit-classic-short-alarm-993.wav"
        pygame.mixer.music.load(self.alarm_sound)

        self.motion_logger = MotionLogger()

        self.label = QLabel('Motion Detection')
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_motion_detection)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_motion_detection)

        self.sensitivity_label = QLabel('Sensitivity')
        self.sensitivity_slider = QSlider()
        self.sensitivity_slider.setOrientation(1)  # Vertical orientation
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(100)

        self.history_list = QListWidget()

        self.set_roi_button = QPushButton('Set ROI', self)
        self.set_roi_button.clicked.connect(self.set_roi)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.sensitivity_label)
        vbox.addWidget(self.sensitivity_slider)
        vbox.addWidget(self.set_roi_button)
        vbox.addWidget(self.history_list)

        self.setLayout(vbox)
        self.show()

    def set_roi(self):
        self.roi_defined = True
        ret, frame = self.cap.read()
        frame = imutils.resize(frame, width=800)

        # Display the frame for the user to draw the ROI
        cv2.namedWindow("Set ROI (Press Enter when done)", cv2.WINDOW_NORMAL)
        cv2.imshow("Set ROI (Press Enter when done)", frame)

        # Define the ROI by clicking and dragging
        self.roi_start = cv2.selectROI("Set ROI (Press Enter when done)", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        # Extract the ROI coordinates
        x, y, w, h = self.roi_start
        self.roi_start = (x, y)
        self.roi_end = (x + w, y + h)

    def start_motion_detection(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to capture a frame.")
                break

            frame = imutils.resize(frame, width=800)

            if self.normal_mode:
                cv2.imshow("Cam", frame)
                if self.alarm_mode:
                    pygame.mixer.music.stop()  # Stop the alarm if switching to normal mode
                    self.alarm_mode = False
            else:
                frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

                if self.roi_defined:
                    # Apply motion detection only within the ROI
                    frame_roi = frame_bw[self.roi_start[1]:self.roi_end[1], self.roi_start[0]:self.roi_end[0]]
                else:
                    frame_roi = frame_bw

                if self.start_frame is None:
                    self.start_frame = frame_roi

                difference = cv2.absdiff(frame_roi, self.start_frame)
                threshold = cv2.threshold(difference, self.sensitivity_slider.value(), 255, cv2.THRESH_BINARY)[1]

                self.start_frame = frame_roi

                cv2.imshow("Cam", threshold)

                if threshold.sum() > 300:
                    print("Motion Detected")
                    self.motion_logger.add_event()
                    notification.notify(
                        title='Motion Detected',
                        message='Motion has been detected!',
                    )

                    if not self.alarm_mode:
                        self.alarm_mode = True
                        pygame.mixer.music.play(-1)  # Play the alarm continuously
                else:
                    if self.alarm_mode:
                        pygame.mixer.music.stop()  # Stop the alarm if no motion is detected
                        self.alarm_mode = False

                # Plot histogram
                # (You can add histogram plotting code here if needed)

            # Update the history list in the GUI
            self.update_history_list()

            key_pressed = cv2.waitKey(30)
            if key_pressed == ord("t"):
                self.normal_mode = not self.normal_mode
                self.start_frame = None
            elif key_pressed == ord("g"):
                pygame.mixer.music.stop()  # Stop the alarm before exiting
                self.motion_logger.export_to_file()  # Export motion event logs
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop_motion_detection(self):
        pygame.mixer.music.stop()  # Stop the alarm
        QApplication.quit()

    def update_history_list(self):
        self.history_list.clear()
        for event in self.motion_logger.motion_events:
            timestamp = event['timestamp']
            item = QListWidgetItem(str(timestamp))
            self.history_list.addItem(item)

if __name__ == '__main__':
    app = QApplication([])
    window = MotionDetectionApp()
    app.exec_()
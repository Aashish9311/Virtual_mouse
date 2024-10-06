import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Camera resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initial variables
previous_x, previous_y = 0, 0

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detect_gesture(landmarks):
    """Detect specific hand gestures based on landmark positions."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate distances
    index_finger_folded = calculate_distance(index_tip.x, index_tip.y, index_mcp.x, index_mcp.y) < 0.05
    middle_finger_folded = calculate_distance(middle_tip.x, middle_tip.y, middle_mcp.x, middle_mcp.y) < 0.05
    thumb_index_pinch = calculate_distance(thumb_tip.x, thumb_tip.y, index_tip.x, index_tip.y) < 0.05

    return index_finger_folded, middle_finger_folded, thumb_index_pinch

def main():
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get hand landmarks
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                # Calculate Index finger tip coordinates
                index_x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                index_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

                # Detect gestures
                index_finger_folded, middle_finger_folded, thumb_index_pinch = detect_gesture(landmarks)

                # Move the mouse based on index finger tip position
                pyautogui.moveTo(index_x, index_y)

                # Perform actions based on the detected gesture
                if index_finger_folded:
                    pyautogui.click()
                elif middle_finger_folded:
                    pyautogui.rightClick()
                elif thumb_index_pinch:
                    pyautogui.doubleClick()

        # Display the image with annotations
        cv2.imshow('Hand Tracking', image)

        # Exit if the 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

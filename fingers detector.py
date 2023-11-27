import cv2


def find_fingers(contour, hull, defects):
    fingers = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate the angle formed by the finger points
        angle = np.degrees(np.arctan2(far[1] - start[1], far[0] - start[0]))

        # Ignore defects that are too close to the wrist or have a small angle
        if d > 10000 and 10 < angle < 100:
            fingers.append(far)

    return fingers

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Threshold the image to get only skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                # Convex hull to get the hand outline
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)

                fingers = find_fingers(contour, hull, defects)

                # Draw contours and fingertips
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
                for finger in fingers:
                    cv2.circle(frame, finger, 5, [0, 0, 255], -1)

                # Detect 4 fingers and display a red point
                if len(fingers) == 4:
                    cv2.circle(frame, (50, 50), 10, [0, 0, 255], -1)

        cv2.imshow('Finger Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

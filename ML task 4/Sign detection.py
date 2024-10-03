# Step 1: Import Libraries
import cv2
import numpy as np
import math

# Step 2: Initialize Camera
cap = cv2.VideoCapture(0)

# Function for trackbar callback (required by OpenCV)
def nothing(x):
    pass

# Step 3: Create a window for trackbars to adjust HSV color space
cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300))

# Create trackbars for color adjustment
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

while True:
    # Step 4: Capture the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Flip the frame horizontally for natural hand orientation
    frame = cv2.flip(frame, 2)
    frame = cv2.resize(frame, (600, 500))

    # Step 5: Define the region of interest (where your hand will be placed)
    cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
    crop_image = frame[1:500, 0:300]

    # Step 6: Convert the image to HSV color space
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    # Get values from the trackbars
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

    # Step 7: Create a mask based on the HSV range from the trackbars
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Filter the mask with the original image
    filtered = cv2.bitwise_and(crop_image, crop_image, mask=mask)

    # Step 8: Perform thresholding and morphological operations
    mask_inv = cv2.bitwise_not(mask)
    threshold_value = cv2.getTrackbarPos("Thresh", "Color Adjustments")
    _, thresh = cv2.threshold(mask_inv, threshold_value, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, (3, 3), iterations=6)

    # Step 9: Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the maximum area (the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to reduce the number of points
        epsilon = 0.001 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # Draw the contours
        hull = cv2.convexHull(approx)  # Use the approximated contour
        cv2.drawContours(crop_image, [max_contour], -1, (50, 50, 150), 2)
        cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

        # Step 10: Find convexity defects
        if len(approx) >= 3:
            hull_indices = cv2.convexHull(approx, returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 0:
                hull_indices = hull_indices.flatten()  # Flatten to 1D array
                hull_indices.sort()  # Ensure indices are sorted
                defects = cv2.convexityDefects(approx, hull_indices)

                count_defects = 0

                # Process each convexity defect
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])

                        # Apply the cosine rule to find the angle
                        a = math.dist(end, start)
                        b = math.dist(far, start)
                        c = math.dist(end, far)
                        angle = (math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180) / math.pi

                        # If the angle is less than 90 degrees, it's likely a finger fold
                        if angle <= 90:
                            count_defects += 1
                            cv2.circle(crop_image, far, 5, [0, 0, 255], -1)

                        # Draw lines around the hand
                        cv2.line(crop_image, start, end, [0, 255, 0], 2)

                # Display number of fingers detected
                finger_count = count_defects + 1  # Count defects and add 1 for the palm
                if finger_count > 5:
                    finger_count = 5  # Cap at 5 fingers
                cv2.putText(frame, f"{finger_count} Finger(s)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Step 11: Show the frame and the mask
    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Filtered Image", filtered)
    cv2.imshow("Frame", frame)

    # Exit on pressing the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Step 12: Release resources
cap.release()
cv2.destroyAllWindows()

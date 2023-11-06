import cv2
import numpy as np
import math
from scipy import stats
def view_image(i):
    cv2.imshow('view', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
i = cv2.imread("subway.jpg")

i_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) 

sobelx = cv2.Sobel(i_gray, cv2.CV_64F, 1, 0) 
abs_sobelx = np.absolute(sobelx)

sobely = cv2.Sobel(i_gray, cv2.CV_64F, 0, 1) 
abs_sobely = np.absolute(sobely)

magnitude = np.sqrt(sobelx**2 + sobely**2) 

edges = cv2.Canny(i_gray, 200, 250) 

lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi/180.0, threshold=20, minLineLength=10, maxLineGap=5)
new_i = i.copy()
for l in lines:
    x1, y1, x2, y2 = l[0]
    cv2.line(new_i, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

circles = cv2.HoughCircles(i_gray, method=cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=150, param2=40, minRadius=15, maxRadius=25)
newer_i = i_gray.copy()
for x, y, r in circles[0]:
    cv2.circle(newer_i, (int(x), int(y)), int(r), (0, 0, 255), thickness=1)

i_blurred = cv2.GaussianBlur(i_gray, ksize = (21,21), sigmaX=0)

circles2 = cv2.HoughCircles(i_blurred, method=cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=150, param2=40, minRadius=15, maxRadius=25)
newer_i = i.copy()
for x, y, r in circles2[0]:
    cv2.circle(newer_i, (int(x), int(y)), int(r), (0, 0, 255), thickness=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
    upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_upperbody.xml") 
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(i_gray, 10, 350)
    i_blurred = cv2.GaussianBlur(i_gray, ksize = (29,1), sigmaX=0)

    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi/180.0, threshold=1, minLineLength=1, maxLineGap=20)
    faces = face_cascade.detectMultiScale(frame, 1.03, 4)

    # Temporary shoulder detection method taken from external project

    def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=2, search_width = 20):
        y = int(y)
        x = int(x)
        columns = img_gray[y:y+h, int(x-search_width/2):int(x+search_width/2)]
        column_average = columns.mean(axis=1)
        gradient = np.gradient(column_average, 3)
        gradient = np.absolute(gradient) 
        max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] 
        max_values = gradient[max_indicies]
        if(max_values.sum() < top_values_to_consider): return None 
        weighted_indicies = (max_indicies * max_values)
        weighted_average_index = weighted_indicies.sum() / max_values.sum()
        index = int(weighted_average_index)
        index = y + index
        return index

    def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
        x_face, y_face, w_face, h_face = face 

        w = int(x_scale * w_face)
        h = int(y_scale * h_face)
        y = y_face + h_face * 5/4 
        if(direction == "right"): x = x_face + w_face - w / 20 
        if(direction == "left"): x = x_face - w + w/20 
        rectangle = (x, y, w, h)

        x_positions = []
        y_positions = []
        for delta_x in range(w):
            this_x = x + delta_x
            this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h)
            if(this_y is None): continue 
            x_positions.append(this_x)
            y_positions.append(this_y)

        lines = []
        for index in range(len(x_positions)):
            lines.append((x_positions[index], y_positions[index]))

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions)
            line_y0 = int(x_positions[0] * slope + intercept)
            line_y1 = int(x_positions[-1] * slope + intercept)
            line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)]
        except(ValueError):
            return None

        value = np.array([line[0][1], line[1][1]]).mean()

        return line, lines, rectangle, value
    # Temporary borrowed code ends here
    left_shoulder = detect_shoulder(grayscale, faces[0], "left")
    right_shoulder = detect_shoulder(grayscale, faces[0], "right")
    if left_shoulder == None or right_shoulder == None: continue
    left_shoulder_rectangle = left_shoulder[2]
    right_shoulder_rectangle = right_shoulder[2]
    left_shoulder_x, left_shoulder_y = left_shoulder[2][0], left_shoulder[2][1] 
    right_shoulder_x, right_shoulder_y = right_shoulder[2][0], right_shoulder[2][1] 

    reference_line_angle = math.atan2(right_shoulder_y - left_shoulder_y, right_shoulder_x - left_shoulder_x)

    reference_line_angle_degrees = math.degrees(reference_line_angle)

    slouch_threshold = 10  
    print(reference_line_angle_degrees)
    if abs(reference_line_angle_degrees) > slouch_threshold:
        print("Slouching shoulders detected")
    else:
        print("Good posture")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    print(left_shoulder_rectangle)
    cv2.rectangle(frame, (int(left_shoulder_rectangle[0]), int(left_shoulder_rectangle[1])), (int(left_shoulder_rectangle[0]) + left_shoulder_rectangle[2], int(left_shoulder_rectangle[1]) + left_shoulder_rectangle[3]), (0, 255, 0), 5)
    cv2.rectangle(frame, (int(right_shoulder_rectangle[0]), int(right_shoulder_rectangle[1])), (int(right_shoulder_rectangle[0]) + right_shoulder_rectangle[2], int(right_shoulder_rectangle[1]) + right_shoulder_rectangle[3]), (0, 255, 0), 5)

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 1:
        break
cap.release()
cv2.destroyAllWindows()

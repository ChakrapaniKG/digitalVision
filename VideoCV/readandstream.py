import cv2

#use Source = 0 for webcam as the input otherwise pass in the path
source = '../race_car.mp4'

#Creating video capture object from the VideoCapture class
video_cap = cv2.VideoCapture(source)

#Creating the presentation window for display of the video being read
win_name = "Video Read..."
cv2.namedWindow(win_name)

#Write the while loop to read and display the video frames.

while True:
    has_frame,frame = video_cap.read()
    if not has_frame:
        break
    cv2.imshow(win_name,frame)

    #Using the wait key to monitor the keyboard for user input
    #key = cv2.waitKey(0) # Display the window indefinately untill user input.
    #key = cv2.waitKey(1) # Displays the window for 1 milli sec
    key = cv2.waitKey(100)

    if key == ord('Q') or key == ord('q') or key == 27:
        break

video_cap.release()
cv2.destroyWindow(win_name)

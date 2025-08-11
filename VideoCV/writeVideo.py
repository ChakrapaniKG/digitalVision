import cv2
import numpy as np

#Reading the video from source

source = '..\\race_car.mp4'

#Creating teh video capture object

video_cap = cv2.VideoCapture(source)

if not video_cap.isOpened():
    print("Error opening the video")

#Display first frame of the video

ret, frame = video_cap.read()
cv2.imshow('First Frame',frame)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#Retrieving the Video Parameters

frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

#Creating the values for fourcc

fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')

#Specify the output format

file_out_avi = 'video_out.avi'
file_out_mp4 = 'video_out.mp4'

#Tweaking the fps

frame_fps = int(fps/3)

#Create the video writer object

out_avi = cv2.VideoWriter(file_out_avi,fourcc_avi,frame_fps,(frame_w,frame_h))
out_mp4 = cv2.VideoWriter(file_out_mp4,fourcc_mp4,frame_fps,(frame_w,frame_h))

#Convinence function for annotating video frames
def drawBannerText(frame,text,banner_height_percent=0.05,text_color=(0,255,0)):
    #Draw a dark banner on top of the image frame
    #Percent : Percentage of banner height from total height
    banner_height = int(banner_height_percent*frame.shape[0])
    cv2.rectangle(frame,(0,0),(frame.shape[1],banner_height),(0,0,0),thickness=-1)
    #Draw text on banner
    left_offset = 20
    location = (left_offset,int(10  + (banner_height_percent * frame.shape[0])/2))
    #Value in the second parameter defines the how much from top the text should be displayed
    #If you use 100 the text will start from 100 th pixel from the top along y-axis (h,w) of an array
    fontscale = 1.5
    fontthickness = 2
    cv2.putText(frame,text,location,cv2.FONT_HERSHEY_PLAIN,fontscale,text_color,fontthickness,cv2.LINE_AA)

frame_count = 0

while True:
    ok,frame = video_cap.read()
    if not ok:
        break
    frame_count += 1
    drawBannerText(frame,'Frame :' + str(int(frame_count)) + ' FPS : '+str(int(frame_fps)))
    out_avi.write(frame)
    out_mp4.write(frame)

video_cap.release()
out_avi.release()
out_mp4.release()
import logging
import sys
from tkinter import Tk, messagebox
import eel
import base64
import time
import os
import json
import math
import cv2
import numpy as np
from camera import VideoCamera

# Global Variables 
shape=""
color=""

# Variables for counting
green2x8 = 0
Grey16 = 0
lightgray2x4 = 0
lime2x2 = 0
lime2x4 = 0
mediumazure4x6 = 0
mediumblue2x4 = 0
orange2x2 = 0
orange2x4 = 0
yellow2x2 = 0
yellow2x3 = 0
yellow2x4 = 0
Count_Indiv_new = [0,0,0,0,0,0,0,0,0,0,0,0]
Count_Indiv_old = [0,0,0,0,0,0,0,0,0,0,0,0]
Count_Running_Total = [0,0,0,0,0,0,0,0,0,0,0,0]
Centroids_new = []
Centroids_old = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
new_arrangement = False

release = False

# Variables to check user inputs for video controls
show_centroid = False
show_boundingbox = False
show_area = False
show_label = False
show_mean_val = False
show_aspect_ratio = False
show_block_count = False
show_gray = False
show_blue = False
show_green = False
show_red = False
show_h = False
show_s = False
show_v = False
show_edge = False
show_circle = False

# to segment the background
lower_bg = np.array([137,137,137])
higher_bg = np.array([255,255,255])

# Set name of Video file to open. Leave name "" to open camera
video_name = "./web/image/car.mp4"
# video_name = ""

x = VideoCamera(video_name)

# Read Images
img = cv2.imread("./web/image/empty.png",cv2.IMREAD_GRAYSCALE)

# send images to load images on GUI
def loadimages():
  img_green2x8 = cv2.imread("./web/image/RawBricksImages/green2x8.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_Grey16 = cv2.imread("./web/image/RawBricksImages/Grey16.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_lightgray2x4 = cv2.imread("./web/image/RawBricksImages/lightgray2x4.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_lime2x2 = cv2.imread("./web/image/RawBricksImages/lime2x2.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_lime2x4 = cv2.imread("./web/image/RawBricksImages/lime2x4.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_mediumazure4x6 = cv2.imread("./web/image/RawBricksImages/mediumazure4x6.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_mediumblue2x4 = cv2.imread("./web/image/RawBricksImages/mediumblue2x4.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_orange2x2 = cv2.imread("./web/image/RawBricksImages/orange2x2.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_orange2x4 = cv2.imread("./web/image/RawBricksImages/orange2x4.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_yellow2x2 = cv2.imread("./web/image/RawBricksImages/yellow2x2.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_yellow2x3 = cv2.imread("./web/image/RawBricksImages/yellow2x3.png",cv2.IMREAD_REDUCED_COLOR_4)
  img_yellow2x4 = cv2.imread("./web/image/RawBricksImages/yellow2x4.png",cv2.IMREAD_REDUCED_COLOR_4)

  img_send_to_js(img_green2x8, "img_green2x8")
  img_send_to_js(img_Grey16, "img_Grey16")
  img_send_to_js(img_lightgray2x4, "img_lightgray2x4")
  img_send_to_js(img_lime2x2, "img_lime2x2")
  img_send_to_js(img_lime2x4, "img_lime2x4")
  img_send_to_js(img_mediumazure4x6, "img_mediumazure4x6")
  img_send_to_js(img_mediumblue2x4, "img_mediumblue2x4")
  img_send_to_js(img_orange2x2, "img_orange2x2")
  img_send_to_js(img_orange2x4, "img_orange2x4")
  img_send_to_js(img_yellow2x2, "img_yellow2x2")
  img_send_to_js(img_yellow2x3, "img_yellow2x3")
  img_send_to_js(img_yellow2x4, "img_yellow2x4")
  
def updatevideo(option):
  global x, video_name
    
  video_name="./web/image/SampleVideos/"+str(option)+".mp4"
  x = VideoCamera(video_name)

# Setup the images to display in html file
@eel.expose
def setup():
  text_send_to_js("Select the Option before pressing Start Video ", "p1")
  img_send_to_js(img, "output")
   
  loadimages()
 
#  Your code depend on image processing
# This is a sample code to change 
# and send processed image to JavaScript  
@eel.expose
def video_feed():                                    # When start button is pressed
  
  global Count_Indiv_new, Count_Indiv_old, Count_Running_Total, Centroids_new, Centroids_old, green2x8, Grey16, lightgray2x4, lime2x2, lime2x4, mediumazure4x6, mediumblue2x4, orange2x2, orange2x4, yellow2x2, yellow2x3, yellow2x4, release, shape, color

  Count_Indiv_new = [0,0,0,0,0,0,0,0,0,0,0,0]
  Count_Indiv_old = [0,0,0,0,0,0,0,0,0,0,0,0]
  Count_Running_Total = [0,0,0,0,0,0,0,0,0,0,0,0]
  Centroids_new = []
  Centroids_old = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
  release = False
  green2x8 = 0
  Grey16 = 0
  lightgray2x4 = 0
  lime2x2 = 0
  lime2x4 = 0
  mediumazure4x6 = 0
  mediumblue2x4 = 0
  orange2x2 = 0
  orange2x4 = 0
  yellow2x2 = 0
  yellow2x3 = 0
  yellow2x4 = 0
  shape=""
  color=""

  option= eel.get_Option()()
  updatevideo(option)
  y = process(x)

  text_send_to_js("Video Started", "p2")
  for image in y:
  #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    img_send_to_js(image, "output")

# Get Camera from video feed
# Add ur codes to process here
def process(camera):
  global Count_Indiv_new, Count_Indiv_old, Count_Running_Total, Centroids_new, Centroids_old, green2x8, Grey16, lightgray2x4, lime2x2, lime2x4, mediumazure4x6, mediumblue2x4, orange2x2, orange2x4, yellow2x2, yellow2x3, yellow2x4, shape, color 

  option= eel.get_Option()()
  text_send_to_js("Option Selected: " + str(option), "p1")

  while True:
    if (release == False):
      
      success, frame = camera.get_frame()
      if(success==False):
        text_send_to_js("Video has Ended", "p2")
        break
      else:

        image = frame.copy()
        grayimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # image pre-processing
        blur = cv2.medianBlur(image,5)

        # Get a mask containing the background
        mask = cv2.inRange(blur, lower_bg, higher_bg)

        # invert mask to get mask containing the blocks
        invert = cv2.bitwise_not(mask)
        
        # Morphological Transformation
        element = cv2.getStructuringElement(cv2.MORPH_OPEN,(5,5))

        # Opening (Remove noise)
        opening_erode = cv2.erode(invert, element, iterations=1)
        opening_dilate = cv2.dilate(opening_erode, element, iterations=1)

        # Closing (Patch small holes in mask)
        closing_dilate = cv2.dilate(opening_dilate, element, iterations=1)
        closing_erode = cv2.erode(closing_dilate, element, iterations=1)

        res = cv2.bitwise_and(image,image, mask= closing_erode)

        #CONTOUR DRAWING
        greyres = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(greyres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Reset count for current frame
        count = 0

        # Identify Blocks
        for cnt in contours:


          objmask = np.zeros(grayimage.shape,np.uint8)
          cv2.drawContours(objmask,[cnt],0,255,-1)
          pixelpoints = cv2.findNonZero(mask)

          area = cv2.contourArea(cnt)
          if (area > 2000 and area<23000):                     #Contour is big enough to be a block
            

            shape=""
            color=""

            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])       # x coordinate for center of contour
            cy = int(M['m01']/M['m00'])       # y coordinate for center of contour
            
            # x,y,w,h = cv2.boundingRect(cnt)
            ## Straight Bounding Rectangle
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            # aspect_ratio = float(w)/h

            ## Rotated Bounding Rectangle
            rect = cv2.minAreaRect(cnt)
            (x, y), (width, height), angle = rect
            aspect_ratio = min(width, height) / max(width, height)
            
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # add centroid of this block into array to keep track of centroid
            Centroids_new.append([cx,cy])
            
            
            mean_val = cv2.mean(image, objmask)
            mean_val = tuple([float("{0:.1f}".format(n)) for n in mean_val])

            if(aspect_ratio>0.81 and aspect_ratio<=1 and area<5600):   
              shape = "2x2"

              if(mean_val[0]>55 and mean_val[0]<90 and mean_val[1]>115 and mean_val[1]<150 and mean_val[2]>105 and mean_val[2]<140):
                color="lime"

              if(mean_val[0]>40 and mean_val[0]<75 and mean_val[1]>80 and mean_val[1]<110 and mean_val[2]>155 and mean_val[2]<195):
                color="orange"

              if(mean_val[0]>55 and mean_val[0]<80 and mean_val[1]>127 and mean_val[1]<160 and mean_val[2]>155 and mean_val[2]<190):
                color="yellow"

            elif(area<8000 and aspect_ratio>0.647 and aspect_ratio<0.849):    
              shape = "2x3"

              if(mean_val[0]>55 and mean_val[0]<80 and mean_val[1]>127 and mean_val[1]<160 and mean_val[2]>155 and mean_val[2]<190):
                color="yellow"
          
            elif(area<11000 and aspect_ratio>0.45 and aspect_ratio<0.647):
              shape = "2x4"

              if(mean_val[0]>90 and mean_val[0]<127 and mean_val[1]>100 and mean_val[1]<127 and mean_val[2]>100 and mean_val[2]<130):
                color="lightgray"

              if(mean_val[0]>110 and mean_val[0]<155 and mean_val[1]>90 and mean_val[1]<120 and mean_val[2]>73 and mean_val[2]<105):
                color="mediumblue"

              if(mean_val[0]>55 and mean_val[0]<90 and mean_val[1]>115 and mean_val[1]<150 and mean_val[2]>105 and mean_val[2]<140):
                color="lime"

              if(mean_val[0]>40 and mean_val[0]<75 and mean_val[1]>80 and mean_val[1]<110 and mean_val[2]>155 and mean_val[2]<195):
                color="orange"

              if(mean_val[0]>55 and mean_val[0]<80 and mean_val[1]>127 and mean_val[1]<160 and mean_val[2]>155 and mean_val[2]<190):
                color="yellow"

            elif(area<18000 and aspect_ratio>0.25 and aspect_ratio<0.32):
              if(mean_val[0]>60 and mean_val[0]<80 and mean_val[1]>90 and mean_val[1]<110 and mean_val[2]>55 and mean_val[2]<70):
                color="green"
                shape = "2x8"
              else:
                color="Grey"
                shape = "16"
              
              # if(mean_val[0]>60 and mean_val[0]<70 and mean_val[1]>60 and mean_val[1]<70 and mean_val[2]>60 and mean_val[2]<70):
                # color="Grey"
                # shape = "16"
              
            elif(area<23000 and aspect_ratio>0.65 and aspect_ratio<0.72):
              shape = "4x6"
              color="mediumazure"
            
            label = color + shape
          
            #update count variable for each type of brick
            if(label == "green2x8"):
              green2x8 += 1
              count += 1
            elif(label == "Grey16"):
              Grey16 += 1
              count += 1
            elif(label == "lightgray2x4"):
              lightgray2x4 += 1
              count += 1
            elif(label == "lime2x2"):
              lime2x2 += 1
              count += 1
            elif(label == "lime2x4"):
              lime2x4 += 1
              count += 1
            elif(label == "mediumazure4x6"):
              mediumazure4x6 += 1
              count += 1
            elif(label == "mediumblue2x4"):
              mediumblue2x4 += 1
              count += 1
            elif(label == "orange2x2"):
              orange2x2 += 1
              count += 1
            elif(label == "orange2x4"):
              orange2x4 += 1
              count += 1
            elif(label == "yellow2x2"):
              yellow2x2 += 1
              count += 1
            elif(label == "yellow2x3"):
              yellow2x3 += 1
              count += 1
            elif(label == "yellow2x4"):
              yellow2x4 += 1
              count += 1

            count_text = "No. Blocks in frame = " + str(count)

            mean_val_display = str(mean_val[0]) + ',' + str(mean_val[1]) + ',' + str(mean_val[2]) 

            aspect_ratio_display = float("{0:.2f}".format(aspect_ratio))

            
            if show_centroid:
              cv2.circle(image,(cx,cy), 2, (0,0,255), 4)

            if show_boundingbox:
              cv2.drawContours(image,[box],0,color=(0,0,255),thickness=2)    

            if show_area:
              cv2.putText(image, str(area), (cx+30, cy+70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(255,0,255), thickness=2)


            if show_aspect_ratio:
              cv2.putText(image, str(aspect_ratio_display), (cx+30, cy+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(255,102,102), thickness=2) 

            if show_mean_val:
              cv2.putText(image, str(mean_val_display), (cx+30, cy+50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(76,0,153), thickness=2)
            
            if show_label:
              cv2.putText(image, label, (cx, cy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,0), thickness=2)
            


        # Counting
        global Count_Indiv_old, Count_Indiv_new
        if (Count_Indiv_new != [0,0,0,0,0,0,0,0,0,0,0,0]):
          Count_Indiv_old =  Count_Indiv_new              #update count for previousframe 

        # update count for currentframe  
        Count_Indiv_new = [green2x8, Grey16, lightgray2x4, lime2x2, lime2x4, mediumazure4x6, mediumblue2x4, orange2x2, orange2x4, yellow2x2, yellow2x3, yellow2x4]
        
        # Sort centroids in current frame to get order of centroids from lowest to highest
        Centroids_new.sort()        

        # Check if there is a change in arrangement / new set of blocks
        if (Count_Indiv_old == Count_Indiv_new):
          global new_arrangement
          for x in range(len(Centroids_new)):
              
            isclose_x = math.isclose(Centroids_old[x][0], Centroids_new[x][0], abs_tol = 3) # New centroid x-coordinate (current frame) close to old centroid x-coordinate (previous frame)

            isclose_y = math.isclose(Centroids_old[x][1], Centroids_new[x][1], abs_tol = 3) # New centroid y-coordinate (current frame) close to old centroid y-coordinate (previous frame)
      
            if ((isclose_x == False) or (isclose_y == False)):
              new_arrangement = True

        # Update Counts on GUI (Specific Color and Total Count)
        
        # If there is one block detected and either (count for current frame is different from count for previous frame) or (new set of blocks comes in)
        

        if ((Count_Indiv_new != [0,0,0,0,0,0,0,0,0,0,0,0]) and ((Count_Indiv_old != Count_Indiv_new) or (new_arrangement == True))):
          global Count_Running_Total

          # Update running count of blocks for each colour
          Count_Running_Total = np.add(Count_Running_Total, Count_Indiv_new)
          # Update Count in GUI for blocks of each colour
          for n in range(len(Count_Running_Total)):
            
            countid = "count_"+str(n)
            text_send_to_js(str(Count_Running_Total[n]), countid)
          
          # Update Count in GUI for total blocks
          totalcount = sum(Count_Running_Total)

          text_send_to_js(str(totalcount), "count_total")
              
        # Update previous frame centroid array
        Centroids_old = Centroids_new
        
        # Reset Centroids_new to contain centroids for next frame
        Centroids_new = []
        new_arrangement = False

        # Reset Counts for each colour for next frame
        green2x8 = 0
        Grey16 = 0
        lightgray2x4 = 0
        lime2x2 = 0
        lime2x4 = 0
        mediumazure4x6 = 0
        mediumblue2x4 = 0
        orange2x2 = 0
        orange2x4 = 0
        yellow2x2 = 0
        yellow2x3 = 0
        yellow2x4 = 0

        imgBlue, imgGreen, imgRed = cv2.split(frame)
        hsvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgValue = cv2.split(hsvimage)

        # Check video control 
        if show_block_count:
          cv2.putText(image, count_text, (10, 35), fontFace=cv2.FONT_ITALIC, fontScale=0.7,color=(0,0,0), thickness=2)
        if show_gray:
          image = grayimage
        if show_blue:
          image = imgBlue
        if show_green:
          image = imgGreen
        if show_red:
          image = imgRed
        if show_h:
          image = imgHue
        if show_s:
          image = imgSaturation
        if show_v:
          image = imgValue

        if show_edge:
          gaussianblur = cv2.GaussianBlur(frame,(5,5),0)
          imEdge = cv2.Canny(gaussianblur,30,100, apertureSize = 3, L2gradient=True)
          image = imEdge
          
        if show_circle:
          gaussianblur = cv2.GaussianBlur(grayimage,(5,5),0)
          detected_circles = cv2.HoughCircles(gaussianblur, cv2.HOUGH_GRADIENT, 1, 25, param1 = 20, param2 = 10, minRadius = 5, maxRadius = 5) 
          if detected_circles is not None: 
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
          
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                # Draw the circumference of the circle. 
                cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(image, (a, b), 1, (0, 0, 255), 3) 

        yield image  
    else:
      
      break

#  Your code depend on image processing
# This is a sample code to change to change 
# the text and send the text to JavaScript
@eel.expose
def addValue():
  val=int(eel.get_Value("p5")())
  val = val + 1
  text_send_to_js(str(val), "p5")

# Stop Video Caturing
# Do not touch
@eel.expose
def stop_video_feed():                        # When stop button is pressed
  global x, release
  x.__del__()
  release = True
  text_send_to_js("Video Stopped", "p2")
  
# Restart Video Capturing
# Do not touch
# @eel.expose
# def restart_video_feed():                       # When restart button is pressed
#   global x, release
#   release = False
#   # x.restart_capturing()
#   text_send_to_js("Video Started", "p2")
#   option = eel.get_Option()()
#   if option:
#     video_name = "./web/image/" + str(option) + ".mp4"
#     x = VideoCamera(video_name)
#     y = process(x)

@eel.expose
def MarkCentroid():                        
  global show_centroid
  show_centroid = not show_centroid

@eel.expose
def BoundingBox():                        
  global show_boundingbox
  show_boundingbox = not show_boundingbox

@eel.expose
def ShowArea():                        
  global show_area
  show_area = not show_area

@eel.expose
def ShowLabel():                        
  global show_label
  show_label = not show_label

@eel.expose
def ShowMeanValue():                        
  global show_mean_val
  show_mean_val = not show_mean_val

@eel.expose
def ShowAspectRatio():                        
  global show_aspect_ratio
  show_aspect_ratio = not show_aspect_ratio

@eel.expose
def ShowBlockCount():                        
  global show_block_count
  show_block_count = not show_block_count

@eel.expose
def ShowGray():                        
  global show_gray
  show_gray = not show_gray

@eel.expose
def ShowBlue():                        
  global show_blue
  show_blue = not show_blue

@eel.expose
def ShowGreen():                        
  global show_green
  show_green = not show_green

@eel.expose
def ShowRed():                        
  global show_red
  show_red = not show_red

@eel.expose
def ShowHue():                        
  global show_h
  show_h = not show_h

@eel.expose
def ShowSaturation():                        
  global show_s
  show_s = not show_s

@eel.expose
def ShowValue():                        
  global show_v
  show_v = not show_v

@eel.expose
def ShowEdge():                        
  global show_edge
  show_edge = not show_edge

@eel.expose
def ShowCircle():                        
  global show_circle
  show_circle = not show_circle

# Send text from python to Javascript 
# Do not touch
def text_send_to_js(val,id):
  eel.updateTextSrc(val,id)()

# Send image from python to Javascript 
# Do not touch
def img_send_to_js(img, id):
  if np.shape(img) == () :
    
    eel.updateImageSrc("", id)()
  else:
    ret, jpeg = cv2.imencode(".jpg",img)
    jpeg.tobytes()
    blob = base64.b64encode(jpeg) 
    blob = blob.decode("utf-8")
    eel.updateImageSrc(blob, id)()

# Start function for app
# Do not touch
def start_app():
  try:
    start_html_page = 'index.html'
    eel.init('web')
    logging.info("App Started")
    eel.start('index.html', size=(1000, 800))

  except Exception as e:
    err_msg = 'Could not launch a local server'
    logging.error('{}\n{}'.format(err_msg, e.args))
    show_error(title='Failed to initialise server', msg=err_msg)
    logging.info('Closing App')
    sys.exit()

if __name__ == "__main__":
  start_app()
  
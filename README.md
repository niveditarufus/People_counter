# People Counter  
![](intro/demo.gif)

## Introduction  
This repository implements a people-counter, which counts people entering and leaving the building and thereby giving a count of the number of people inside. This is written in written in python and performs real-time.

### The algorithm  
#### Detection Phase  
The model used for detection is MobileNet SSD.  
In this phase we run the object tracker to:  
1. Detect new objects that have entered the view.  
2. Check if any of the existing objects "disappeared" durinf the tracking phase.
For each detected object we create or update an object tracker with the new bounding box coordinates. The detection phase is run only once in every N frames at it will be very expensive computationally.  

#### Tracking Phase
In this phase, we create an object tracker to track objects as they move in the frame. The tracking will continue until we’ve reached the N-th frame.  
The entire process repeats again.

### Dependencies  
dlib  
imutils  
numpy  
opencv-python  
scipy  

### Quick Start

1. Clone the repository:  
`git clone https://github.com/niveditarufus/People_counter.git`  
2. run:  
`cd People_counter`
3. Install all dependencies required, run:  
`pip3 install -r requirements.txt`  
3. Run Demo  
usage: python3 SimplePeopleCounter.py
				[--protext PATH TO CAFFE 'DEPLOY' PROTEXT FILE]  
				[--model PATH TO CAFFE PRE-TRAINED MODEL]       
				[--protext PATH TO CAFFE 'DEPLOY' PROTEXT FILE]  
				[--model PATH TO CAFFE PRE-TRAINED MODEL]  
				[--input PATH TO INPUT VIDEO FILE]  
				[--output PATH TO OUTPUT VIDEO FILE]  
				[--confidence MINIMUM PROBABILITY TO FILTER WEAK DETECTIONS, default = 0.4]  
				[--skip-frames NO. OF FRAMES BETWEEN DETECTIONS, default = 30]  
Run:  
`python3 SimplePeopleCounter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi`




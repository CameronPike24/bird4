# This example demonstrates Object Detection using Tensorflow Lite
# It is based on:
# https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection
#
# cv2 references this file and object_detection/object_detector.py from
# example above replaced with 'self.auto_analyze_resolution'

import time
from kivy.clock import mainthread
from kivy.graphics import Color, Line, Rectangle
from kivy.core.text import Label as CoreLabel
from kivy.metrics import dp, sp
from kivy.utils import platform
import numpy as np
from camera4kivy import Preview
from object_detection.object_detector import ObjectDetector
from object_detection.object_detector import ObjectDetectorOptions
from kivy.clock import Clock
from kivy.app import App
#import datetime as dt


class ClassifyObject(Preview):

      
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classified = []
        
      
        

        
        num_threads = 4
        enable_edgetpu = False    ## Change this for Coral Accererator
        if platform == 'android':
            model = 'object_detection/model.tflite'
        elif enable_edgetpu:
            model = 'object_detection/efficientdet_lite0_edgetpu.tflite'
            
        else:
            model = 'object_detection/efficientdet_lite0.tflite'
            
        options = ObjectDetectorOptions(
            num_threads = 4,
            score_threshold = 0.8,
            #max_results = 3,
            max_results = 1,
            enable_edgetpu = enable_edgetpu)
        self.detector = ObjectDetector(model_path=model, options=options)
        # Get the required analyze resolution from the detector, a 2 ele list.
        # as a concequence, scale will be a 2 ele list
        self.auto_analyze_resolution = self.detector._input_size
        self.start_time = time.time()
        

        
        #Wait until app is initialised before getting running app
        Clock.schedule_once(self.after_init)       
        
         
    def after_init(self, dt):
        #App.get_running_app().root.get_character_selection_screen()    
        #print(App.get_running_app())
        #global get_app
        
        get_app = App.get_running_app()
        #pass
            
        #Set info button is dispalyed to 0
        #self.info_button_is_displayed = 0 
        #Set continue detection to 0
  
                
       
        
        
        

    ####################################
    # Analyze a Frame - NOT on UI Thread
    ####################################

    def analyze_pixels_callback(self, pixels, image_size, image_pos,
                                image_scale, mirror):
                                
        #Counter to determine how many times object was detected                        
        self.detected_counter = 0 
        #Create list for probabilities
        self.probability_list = []
                           
                                
        # Convert pixels to numpy rgb
        rgba = np.fromstring(pixels, np.uint8).reshape(image_size[1],
                                                       image_size[0], 4)
        rgb = rgba[:,:,:3]
        # detect
        detections = self.detector.detect(rgb)
        
       
        
        
        print("at analyze_pixels_callback")
        now = time.time()
        fps = 0
        if now - self.start_time:
            fps = 1 / (now - self.start_time)
        self.start_time = now
        found = []
        for detection in detections:
            # Bounding box, pixels coordinates
            x = detection.bounding_box.left
            y = detection.bounding_box.top
            w = detection.bounding_box.right - x 
            h = detection.bounding_box.bottom - y

            # Map tflite style coordinates to Kivy Preview coordinates
            y = max(image_size[1] -y -h, 0)
            if mirror:
                x = max(image_size[0] -x -w, 0)
                
            # Map Analysis Image coordinates to Preview coordinates
            # image_scale is a list because we used self.auto_analyze_resolution
            x = round(x * image_scale[0] + image_pos[0])
            y = round(y * image_scale[1] + image_pos[1])
            w = round(w * image_scale[0])
            h = round(h * image_scale[1])

            # Category text for canvas
            category = detection.categories[0]
            class_name = category.label
            probability = round(category.score, 2)
            #result_text = class_name +\
                #' (Probability: {:.2f} )'.format(probability)
                
            #result_text = class_name  
            result_text = ''              
                
               
            label = CoreLabel(font_size = sp(20))
            
            
            label.text = result_text
            label.refresh()

            # Thread safe result
            found.append({'x':x, 'y':y, 'w':w, 'h':h, 't': label.texture, 'class_detected': class_name, 'probability_detected': probability})
            
             
        self.make_thread_safe(list(found)) ## A COPY of the list     
               

    @mainthread
    def make_thread_safe(self, found):
        #self.classified = found
        if self.camera_connected:
            self.classified = found
            print("camera is connected at make_thread_safe") 
        else:
            # Clear local state so no thread related ghosts on re-connect
            self.classified = []   
            print("camera is disconnected at make_thread_safe")     

    ################################
    # Canvas Update  - on UI Thread
    ################################
        
    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        # Add the analysis annotations
        #Color(0,1,0,1)
        Color(1,1,1,1)
        
        
        #Color(1, 1, 1,1)
        print("At canvas_instructions_callback")
        
        
        
        for r in self.classified:
            print("for r in self.classified")
            
            #Increment counter each time object is detected
            self.detected_counter = self.detected_counter + 1
        
            self.image_name = r['class_detected']
            self.image_name_plus_ext = self.image_name + '.png'
            print(self.image_name_plus_ext)
            
            self.propability_score = r['probability_detected']
            print("propability_score")
            print(self.propability_score)
        
            self.probability_list.append({'probability_detected_score': self.propability_score, 'class_name': self.image_name})
            
            # Draw box
            Line(rectangle=(r['x'], r['y'], r['w'], r['h']), width = dp(1.5))
            # Draw text
                      
            Rectangle(size = r['t'].size,
                      pos = [r['x'] + dp(10), r['y'] + dp(10)],
                      texture = r['t'])       
       
            print("counter")
            print(self.detected_counter)
            
            #print("info button value")
            #print(self.info_button_is_displayed)
            
            
            #if(self.info_button_is_displayed is False):
            #adjust the self.detected_counter to >2 if the phone has a slow frames per second
            self.info_button_is_displayed = App.get_running_app().continue_detection(self) 
            print('self.info_button_is_displayed in classifyobject')
            print(self.info_button_is_displayed)
       
            if(self.detected_counter > 3): 
                if(self.info_button_is_displayed == 0):    
                
                    print("probability_list") 
                    print(self.probability_list)    
                    
                                      
                    sorted(self.probability_list , key=lambda k: (k['probability_detected_score'] , k['class_name'])) 
                    print("sorted probability_list") 
                    print(self.probability_list) 
                    
                    #next(item for item in self.probability_list if item["name"] == "Pam")
                    #next(item for item in self.probability_list)
                    #print(item)
                    for item in self.probability_list:
                        self.probability_highest_value = item['probability_detected_score']
                        self.class_name_highest_value = item['class_name']
                        #get first result then break loop as we only want first item
                        break
                        
                    print("class name highest value")
                    print(self.class_name_highest_value)
                    
                    print("probabilty highest value")
                    print(self.probability_highest_value)                     
              
                    #Add info button widget to screen
                    App.get_running_app().add_object_info_layout(self.class_name_highest_value,self.probability_highest_value)
                    #self.info_button_is_displayed = 1
                 
            
            
            
            
            
            
            
            
            
            
            
                        


from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty,NumericProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.utils import platform
from classifyobject import ClassifyObject
from kivymd.uix.screen import MDScreen
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.button import MDRaisedButton,MDFlatButton, MDFillRoundFlatIconButton
from kivymd.uix.card import MDCard
from kivy.clock import Clock

import librosa
from kivy_garden.lazyloader import LazyLoader
#from scipy.signal import stft

from kivy.uix.anchorlayout import AnchorLayout

from kivy.uix.popup import Popup



from jnius import autoclass
from audiostream import get_input
import wave
#
import os
from android.permissions import request_permissions,Permission,check_permission
from kivy_garden.graph import Graph, LinePlot, MeshLinePlot
import numpy as np 
from array import array
from datetime import datetime
 
#import tensorflow as tf
import zipfile
import wave, struct
import audioop
import sys
from collections import deque
import time
#import threading
import csv



#class AppLayout(FloatLayout):
class AppLayout(MDScreen):
    detect = ObjectProperty()
    #Window.fullscreen = 'auto'
    change_height = StringProperty(None)
    


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

  
    def on_size(self, layout, size):

        if Window.width < Window.height:
            print("Width < height")
            print(Window.width)
            print(Window.height)            
            self.change_height = '0.87'
        
         
            
        else:
        
            print("Width > height")
            print(Window.width)
            print(Window.height)
            self.change_height = '0.75'

    
    def __del__(self):
        print("AppLayout object destroyed")   
      
        
        
        
class ButtonsLayout(RelativeLayout):
#class ButtonsLayout(Screen):
    normal = StringProperty()
    down = StringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if platform == 'android':
            self.normal = 'icons/camera_white.png'
            self.down   = 'icons/camera_red.png'
        else:
            self.normal = 'icons/camera_white.png'
            self.down   = 'icons/camera_red.png'
  
    def on_size(self, layout, size):
        if platform == 'android': 
            self.ids.screen.min_state_time = 0.5
        else:
            self.ids.screen.min_state_time = 1
        if Window.width < Window.height:
            self.pos = (0.5 , 0.5)
            self.size_hint = (0.3, 0.3)

            self.ids.screen.pos_hint  = {'center_x':.5,'center_y':.5}
            self.ids.screen.size_hint = (.3, .3)
        else:
            self.pos = (Window.width * 0.5, 0.5)
            self.size_hint = (0.3 , .3)

            self.ids.screen.pos_hint  = {'center_x':.5,'center_y':.5}
            self.ids.screen.size_hint = (.3, .3)

    #def screenshot(self):
        #self.parent.detect.capture_screenshot()
        #self.parent.addcamerabuttondetect.detect.capture_screenshot()
        #self.cameradetect.detect.connect_camera(enable_analyze_pixels = True)
        #self.detect.capture_screenshot()
        #self.addcamerabutton.detect.capture_screenshot()
        #self.parent.addcamerabutton.detect.capture_screenshot()
        #self.addcamerabutton.detect.capture_screenshot()
        #self.applayout.detect.capture_screenshot()

    def select_camera(self, facing):
        self.parent.detect.select_camera(facing)
        


class ButtonsLayoutInfo(RelativeLayout):
   
    
    def __init__(self, **kwargs):
        super(ButtonsLayoutInfo, self).__init__(**kwargs)

        self.btn = self.ids['info']
        #pos_hint: {'center_x': 0.1, 'center_y': 0.2}  
        #size_hint: (.5,.1)  
        

    def add_btn(self, *args):   
        lst = []
        #pos_hint: {'center_x': 0.1, 'center_y': 0.2}  
        
        print("window size")
        print(Window.width)
        
        #If the button is hidden then move it into view
        #if(self.btn.y == 5000):
        
        if(self.btn.pos_hint=={'x': 0.2, 'y': 5000}):
            print("yes self.btn.y == 5000")
            self.btn.y = self.saved_y
            self.btn.pos_hint={'x': 0.2, 'y': self.btn.y}
            self.btn.pos_hint={'x': 0.5, 'y': 0.5}
            #size_hint: (.5,.1)   
            
        else:
            
            print("no self.btn.y != 5000")    
            print("self.btn.y after remove")
            print(self.btn.y)  
            print("self.btn.pos_hint after remove")
            print()
            self.btn.pos_hint: {'center_x': 0.5, 'center_y': 0.5}       
        
        for arg in args:
            #self.new_text = arg
            lst.append(arg)
            #print("args")    

        self.detected_object_name = lst[0]
        self.object_all_info = lst[1]
        self.probability_highest_value = lst[2]
        print("add_btn probability")
        print(self.probability_highest_value)
        #self.object_general_info = lst[1]
        #self.object_info_1 = lst[2]
        
        #print("self.object_info_1")
        #print(self.object_info_1)
        #self.btn.text = 
        self.detected_object_name = self.detected_object_name.replace("-", " ")        
        self.btn.text = self.detected_object_name.title()
        #print("we changed the text")
        


    def remove_btn(self, *args):

        
        #self.btn.background.opacity: 0
        #self.btn.background_color: (0, 0, 0, 0) 
        #self.btn.size: (0,0)
        #self.size_hint = (0,0) 
        
        
        
        #self.btn.text = ''
        #self.btn.opacity: 0
        #self.btn.md_bg_color: [0,0,0,0]
        self.btn.disabled: True # To make sure it doesn't capture events accidentally
   
        self.saved_y = self.btn.y
        print("self.btn.y")
        print(self.btn.y)      
       
        
        
        # Now move the widget offscreen (recall that pos is just an alias for (x, y))
        self.btn.y = 5000
        print("changed self.btn.y")
        print(self.btn.y)
        
        self.btn.pos_hint ={'x':.2, 'y':self.btn.y}
        print("self.btn.pos_hint")
        print(self.btn.pos_hint)
        #self.saved_y = self.btn.y



 
 
class ButtonLayoutObjectDetection(RelativeLayout):
    
    
    def __init__(self, **kwargs):
        super(ButtonLayoutObjectDetection, self).__init__(**kwargs)

        self.btn = self.ids['info']
        ########################################################################################
        ########To note this creates a start button to start the object detection process#######
        ########################################################################################
        
        
        
        #pos_hint: {'center_x': 0.1, 'center_y': 0.2}  
        #size_hint: (.5,.1)  
        

    #def add_btn(self, *args): 
    def add_btn(self,info_posted):        

        self.btn.pos_hint: {'center_x': 0.5, 'center_y': 0.5}        
        self.btn.text = info_posted
        
        

                
        
        
        
      
    def show_more_info(self):
        
        
        #print("we at show more info")  
        #print("self.object_info_1")
        #print(self.object_info_1) 
        
        #print("we at show more info")  
        #print("self.object_general_info")
        #print(self.object_general_info)
        
            
        #self.btn.text = self.object_general_info
        self.btn.text = self.object_all_info
        self.size_hint = (1,1) 
        
        
        
    def remove_btn(self, *args):

        
        #self.btn.background.opacity: 0
        #self.btn.background_color: (0, 0, 0, 0) 
        #self.btn.size: (0,0)
        #self.size_hint = (0,0) 
        
        
        
        #self.btn.text = ''
        #self.btn.opacity: 0
        #self.btn.md_bg_color: [0,0,0,0]
        self.btn.disabled: True # To make sure it doesn't capture events accidentally
   
        self.saved_y = self.btn.y
        print("self.btn.y")
        print(self.btn.y)      
       
        
        
        # Now move the widget offscreen (recall that pos is just an alias for (x, y))
        self.btn.y = 5000
        print("changed self.btn.y")
        print(self.btn.y)
        
        self.btn.pos_hint ={'x':.2, 'y':self.btn.y}
        print("self.btn.pos_hint")
        print(self.btn.pos_hint)
        #self.saved_y = self.btn.y
        
        
        
        

class ButtonLayoutAudioDetection(RelativeLayout):

    object_all_info = StringProperty()
    detected_object_name = StringProperty()
    object_info_1 = StringProperty()
    object_info_2 = StringProperty()
   
   
    
    def __init__(self, **kwargs):
        super(ButtonLayoutAudioDetection, self).__init__(**kwargs)

        self.btn = self.ids['info']
        #pos_hint: {'center_x': 0.1, 'center_y': 0.2}  
        #size_hint: (.5,.1)  
        

    #def add_btn(self, *args): 
    def add_btn(self,birdclass,x_pos,y_pos,general_info,info1,info2):        

        self.btn.pos_hint: {'center_x': x_pos, 'center_y': y_pos}        
        self.btn.text = birdclass
        self.object_all_info = general_info
        self.detected_object_name = birdclass
        self.object_info_1 = info1
        self.object_info_2 = info2
        #self.birdname = birdclass
        
        
        
        

        
        
    def remove_btn(self, *args):

        
        #self.btn.background.opacity: 0
        #self.btn.background_color: (0, 0, 0, 0) 
        #self.btn.size: (0,0)
        #self.size_hint = (0,0) 
        
        
        
        #self.btn.text = ''
        #self.btn.opacity: 0
        #self.btn.md_bg_color: [0,0,0,0]
        self.btn.disabled: True # To make sure it doesn't capture events accidentally
   
        self.saved_y = self.btn.y
        print("self.btn.y")
        print(self.btn.y)      
       
        
        
        # Now move the widget offscreen (recall that pos is just an alias for (x, y))
        self.btn.y = 5000
        print("changed self.btn.y")
        print(self.btn.y)
        
        self.btn.pos_hint ={'x':.2, 'y':self.btn.y}
        print("self.btn.pos_hint")
        print(self.btn.pos_hint)
        #self.saved_y = self.btn.y
        
        
 
        



class Content(MDCard):

 

    def __init__(self, **kwargs):
        #self.window = Window
        super(Content, self).__init__(**kwargs)
        self.img = self.ids['my_image']
        self.lbl = self.ids['my_label']
        self.my_title = self.ids['my_title']
        self.my_info_1 = self.ids['my_info_1']
        self.my_info_2 = self.ids['my_info_2']      
        

    
    
    
    def add_img_name(self, *args):    

    
        lst1 = []
        print("hello")
        for arg in args:
            #self.new_text = arg
            lst1.append(arg)
            print("args")  
        self.detected_object_name = lst1[0]     
        self.object_all_info = lst1[1]
        self.object_info_1 = lst1[2]
        self.object_info_2 = lst1[3]
       
        
        print(self.detected_object_name)
        print(self.object_all_info)
        self.image_full_path = "images/" + self.detected_object_name + ".png"
        
        self.img.source = self.image_full_path
        self.detected_object_name = self.detected_object_name.replace("-", " ")
        self.my_title.text = self.detected_object_name.title()
        
        self.lbl.text = self.object_all_info
        self.my_info_1.text = self.object_info_1
        self.my_info_2.text = self.object_info_2
        


 
 
class ContentAudio(MDCard):

 

    def __init__(self, **kwargs):
        #self.window = Window
        super(ContentAudio, self).__init__(**kwargs)
        self.img = self.ids['my_image']
        self.lbl = self.ids['my_label']
        self.my_title = self.ids['my_title']
        self.my_info_1 = self.ids['my_info_1']
        self.my_info_2 = self.ids['my_info_2']      
        

    
    
    
    def add_img_name(self, *args):    

    
        lst1 = []
        print("hello")
        for arg in args:
            #self.new_text = arg
            lst1.append(arg)
            print("args")  
        self.detected_object_name = lst1[0]     
        self.object_all_info = lst1[1]
        self.object_info_1 = lst1[2]
        self.object_info_2 = lst1[3]
       
        
        print(self.detected_object_name)
        print(self.object_all_info)
        self.image_full_path = "images/" + self.detected_object_name + ".png"
        
        self.img.source = self.image_full_path
        self.detected_object_name = self.detected_object_name.replace("-", " ")
        self.my_title.text = self.detected_object_name.title()
        
        self.lbl.text = self.object_all_info
        self.my_info_1.text = self.object_info_1
        self.my_info_2.text = self.object_info_2
        

    def remove_audio_dialog(self):        
        #self.remove_dialog_audio = ButtonLayoutAudioDetection()
        
        #Add RecordForm class widget
        self.recorder = MDScreen(name="record_form")     
        self.recorderform = RecordForm()          
        self.recorder.remove_widget(self.recorderform)     
        
        #self.remove_dialog_audio.remove_widget(self.adddialog)
        #self.remove_dialog_audio.remove_widget(self.content_object)
        
        #self.remove_dialog_audio.remove_dialog()   
 
 
 
        

class Settings(MDCard):  

    def __init__(self, **kwargs):
        super(Settings, self).__init__(**kwargs)

      
class Weather(MDCard):  

    def __init__(self, **kwargs):
        super(Weather, self).__init__(**kwargs)  

    pass    
        



PATH = "rec_test1.wav"
 
recordtime = 5
samples_per_second = 60
audio_button_displayed_count = 0
       

class RecordForm(MDScreen):
    #def __init__(self):
    def __init__(self, **kwargs):   
    
        super(RecordForm, self).__init__(**kwargs)    
        #self.plot = MeshLinePlot(color=[1, 0, 0, 1])   
        self.plot = LinePlot(line_width=2,color=[1, 1, 1, 1]) 
     
        # get the needed Java classes
        self.MediaRecorder = autoclass('android.media.MediaRecorder')
        self.AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
        self.AudioFormat = autoclass('android.media.AudioFormat')
        self.AudioRecord = autoclass('android.media.AudioRecord')
        # define our system
        self.SampleRate = 44100
        #self.SampleRate = 16000
        self.ChannelConfig = self.AudioFormat.CHANNEL_IN_MONO
        self.AudioEncoding = self.AudioFormat.ENCODING_PCM_16BIT
        self.BufferSize = self.AudioRecord.getMinBufferSize(self.SampleRate, self.ChannelConfig, self.AudioEncoding)
        #self.outstream = self.FileOutputStream(PATH)
        self.sData = []
        self.copy_sData = []
        #self.mic = get_input(callback=self.mic_callback, source='mic', buffersize=self.BufferSize)
        self.mic = get_input(callback=self.mic_callback, source='default', buffersize=self.BufferSize)
        print("This is the audio source")
        print(self.AudioSource)
        print("This is the mic channels")
        print(self.mic.channels)
        print("self.BufferSize")
        print(self.BufferSize)
        
        self.queue_frames = deque(maxlen=self.BufferSize)
        self.decoded = []
        self.decoded_copy = []
        self.recording_has_started = False
        self.amplitude_high = 0
        self.display_audio_detection_button_active = False
        
        
    
        
        
        ###################################################
        # Step 1 - Load bird model
        ######################################################

        # Load TFLite model and allocate tensors.
        #This is to load the default yamnet model which you dont need to do
        #with open('lite-model_yamnet_tflite_1.tflite', 'rb') as fid:
        
        
        

        try:
            # Import TFLite interpreter from tflite_runtime package if it's available.
            from tflite_runtime.interpreter import Interpreter
            from tflite_runtime.interpreter import load_delegate
            Interpreter_audio = Interpreter
        except ImportError:
            # If not, fallback to use the TFLite interpreter from the full TF package.
            import tensorflow as tf
            #Interpreter_audio = tf.lite.Interpreter
            #load_delegate = tf.lite.experimental.load_delegate        
        
        
        

        #Open the bird model
        with open('object_detection/my_birds_model.tflite', 'rb') as fid:
            tflite_model = fid.read()    

        #interpreter = tf.lite.Interpreter('lite-model_yamnet_tflite_1.tflite')
        #interpreter = tf.lite.Interpreter(model_content=tflite_model)             
        self.interpreter_audio = Interpreter_audio(model_content=tflite_model) 
        
        
        ###################################################
        # Step 2 - Get labels/bird names from metadata inside mdel
        ######################################################

        try:
            #with zipfile.ZipFile('my_birds_model.tflite') as model_with_metadata:
            #with zipfile.ZipFile('./birds_models/my_birds_model.tflite') as model_with_metadata: 
            with zipfile.ZipFile('object_detection/my_birds_model.tflite') as model_with_metadata: 
  
                if not model_with_metadata.namelist():
                    raise ValueError('Invalid TFLite model: no label file found.')


                #the file name below retrieves the default Yamnet model metadata
                #file_name = model_with_metadata.namelist()[0]
                #or
                #the file name below retrieves the new bird model metadata namely [1]
                file_name = model_with_metadata.namelist()[1]

                with model_with_metadata.open(file_name) as label_file:
                    label_list = label_file.read().splitlines()
                    print("label_list")
                    print(label_list)
                    #_label_list = [label.decode('ascii') for label in label_list]
                    self._label_list = [label.decode('utf-8') for label in label_list]
      
                    print("_label_list")
                    print(self._label_list)  #Should print list of bird codes ['azaspi1', 'chcant2', 'houspa', 'redcro', 'wbwwre1']
      
                    print(len(self._label_list))  # should print 5   
      
                    #print(test_data.index_to_label[0])
      
        except zipfile.BadZipFile:
            print('ERROR: Please use models trained with Model Maker or downloaded from TensorFlow Hub.')
            raise ValueError('Invalid TFLite model: no metadata found.')        
        
      
        
        ###################################################
        # Step 3 - Get input/output details of model to ensure the audio sample fits the model input
        ######################################################

        self.input_details = self.interpreter_audio.get_input_details()
        print("input_details")
        print(self.input_details)
        self.waveform_input_index = self.input_details[0]['index']
        self.output_details = self.interpreter_audio.get_output_details()
        print("output_details")
        print(self.output_details)

        #Dont need this it is for the default Yamnet model
        self.scores_output_index = self.output_details[0]['index']
        print("scores_output_index 0")
        print(self.scores_output_index)

        #Get the output scores of the bird model
        self.scores_output_index1 = self.output_details[1]['index']
        print("")
        #Shows 116
        print(self.scores_output_index1)

        #The below is to create a wavfile of silence for testing only
        #Input: 0.975 seconds of silence as mono 16 kHz waveform samples.
        #waveform = np.zeros(int(round(0.975 * 16000)), dtype=np.float32)        
        


    def __del__(self):
        print("RecordForm object destroyed")    

 
    def mic_callback(self, buf):
    
        #######################################################################
        #mic_callback gets polled every few milliseconds so this function runs continuously
        #The main function is to append data to sDatat buffer
        #If the amplitude is high enough and self.recording_has_started == False
        #then this function starts recording and appends the sData buffer. It stops recording after 2 seconds
        ######################################################################
        print("we at mic_callback")
    
        global levels
    
        self.sData.append(buf)
        print ('got : ' + str(len(buf)))
        print(self.sData)
        
        # convert our byte buffer into signed short array
        '''
        data_struct = struct.unpack("<h",buf)
        sound[0] = int(data_struct[0])
        print("sound[0]")
        print(sound[0])
        '''
        
        # assuming myByte is your byte array:
        #len_count = len(buf)/2

        #sh = struct.unpack('h'*len_count , buf)
        #sh = np.array(struct.unpack('<h', buf))
        #print("sh")
        #print(sh)
        
        '''
        if len(levels) >= 100:
            levels = []
            #empty levels
            print("deleted levels")       
        '''
        
        
        #self.mx = np.frombuffer(buf, np.int16)
        #####################################
        #For plotting the graph we get a copy of the buffer 
        #####################################
        
        self.decoded = np.frombuffer(buf, np.int16)
        '''
        print("mx")
        print(self.mx)     
        
        print("length of mx")
        print(len(self.mx))
        
        '''
        
        print("length of self.decoded before downsample")
        print(len(self.decoded))   
        
        print("self.decoded before downsample")
        print(self.decoded)                  
        
        
        #Get every 5th value - downsample
        
        self.decoded = self.decoded[0::7]
        print("length of self.decoded after downsample")
        print(len(self.decoded))   
        
        print("self.decoded after downsample")
        print(self.decoded) 
        
        #Normalise the values between -1 and 1
        #self.decoded = np.round(self.decoded / np.linalg.norm(self.decoded, axis=0),2) 
        amin = -1500
        amax = 1500        

        #self.decoded = [((x - minL) * 2)/(maxL - minL) - 1 for x in self.decoded]
        self.decoded_copy = [round(((x - amin) * 2)/(amax - amin) - 1,2) for x in self.decoded]
        
        #self.decoded_copy = self.decoded
        '''
        #amin, amax = min(self.decoded), max(self.decoded)
        for n, val in enumerate(self.decoded):
            #self.decoded[i] = (val-amin) / (amax-amin)
           
            self.decoded_copy[n] = round(((val - amin) * 2)/(amax - amin) - 1,1)
        '''
              
        print("self.decoded_copy after normalize")
        print(self.decoded_copy)        
        
        
        #Add self.decoded audio to queue ** This is a queue so dont need to empty        
        self.queue_frames = self.decoded_copy
        

        
        
        ##############################################################
        #If amplitude is over 1000 then start recording - this is just for testing and will change
        ##############################################################
        #self.amplitude_high = 101
        self.record_time = 2 # We are filling sData with audio and need to stop in 2 seconds time to create the wave file
        
        self.amplitude_high = max(self.decoded)
        
        #empty self.decoded
        self.decoded = []
        #empty self.decoded_copy
        self.decoded_copy = []        
     
        '''
        
        #Convert to np array to use argmax
        results_np_array_sData = np.array(self.sData)
        result_index_sData = results_np_array_sData.argmax()
        print("result_index_sData")
        print(result_index_sData)
        
       
        print("Highest amplitude value for result_index_sData")
        #This is a 1-dimensional array so get the value with the index
        self.amplitude_high = results_np_array_sData[result_index_sData]
        '''
        
        
        
        
        print("self.amplitude_high")
        print(self.amplitude_high)
        print("self.recording_has_started")
        print(self.recording_has_started)
        
        
        ##########################################################################
        #The first time the program runs self.recording_has_started
        #was set to false which was set at initialization in the RecordForm class
        #When we start recording we set self.recording_has_started = True
        ##########################################################################
        
        if self.amplitude_high > 2000 and self.recording_has_started == False:
        
            print("self.amplitude_high > 2000 AND self.recording_has_started == False")       
            
            #######################################################
            #Set self.recording_has_started = True so the next pass of this if statement we will not call self.stop again
            #######################################################
                
            self.recording_has_started = True
            print("self.recording_has_started")
            print(self.recording_has_started)
            
            ##############################################
            #We are now recording and appensing sData buffer while the mic_callback runs every few milliseconds 
            #We will continue to record for the next 2 seconds (set in self.record_time).
            #Then we will call self.stop that stops the microphone,
            #creates the wave file and does inference looking for an audio match
            ##############################################
            
            Clock.schedule_once(self.stop,self.record_time)
        
            
            
        else:  
        
            #self.recording_has_started == True so do nothing while we are recording
            pass   
            
            
        
        ###############################################################
        #If  self.recording_has_started == True then dont empty the buffer as we are still recording
        #If  self.recording_has_started == False then we are not recording we can empty the buffer
        #or else it fills up
        ###############################################################
          
        if self.recording_has_started == False: 
            print("if self.recording_has_started is False")
            print("we emptying the buffer")
            self.empty_buffer()
            #Delete the copy of sData as well
            self.copy_sData = []
        else:
            pass    
        



        print("self.queue_frames sent to graph")
        print(self.queue_frames)      
        
        print("self.decoded after empty")
        print(self.decoded)  
        
        
        
           
        
        try:
            #call plot graph
            self.get_value(self.queue_frames)                

        except:           

            pass
        
        

    
    

    def start_record(self):
        #############################################################
        #This function is called from the setupkv file
        #When the start record button is pressed this function is called
        #############################################################
        
        #self.b_record.disabled = True
        #self.p_bar.max = recordtime
        #REC.prepare()
        #REC.start()
        self.ids.graph.add_plot(self.plot)
        #Use sample rate
        
        self.start()
        
        #Wait for buffer to fill
        print("waiting for buffer to fill")
        time.sleep(1)             
            
        print("buffer is full starting capture now")            
        
        #time.sleep(5)
        #Clock.schedule_interval(self.get_value, 1/60)
        #Clock.schedule_interval(self.get_value, 1/120)
        #Clock.schedule_interval(self.get_value, 0.005)
        #self.thread.start()
   
        
        #stop recording after recordtime value
        #Clock.schedule_once(self.stop_record, recordtime)
  
 
    #def stop_record(self, dt):
    def stop_record(self):
        #Clock.unschedule(self.update_display)
        #self.p_bar.value = 0
        #REC.stop()
        self.stop()
        #REC.start()
        #Clock.unschedule(self.get_value)
        
        

        
        
        
        
    #def get_value(self, dt):
    #def get_value(self, array_values):
    def get_value(self,concatenated_queue_frames):
        # Do a call to __call__ in Recorder class
        
        
        print("we at def get_value")
         
        #concatenated_queue_frames = REC()    

        print("concatenated_queue_frames")
        print(concatenated_queue_frames)        
        
     
        #NB j//5 not j/5 removes any remainder which results in a float
        #self.plot.points = [(i, j//20) for i, j in enumerate(concatenated_queue_frames)] 
        #self.plot.points = [(i, j//5) for i, j in enumerate(islice(array_values, 300))] 
        self.plot.points = [(i, j) for i, j in enumerate(concatenated_queue_frames)] 
        
        print("self.plot.points")
        print(self.plot.points)
        #self.plot.points = [(i, j/100) for i, j in enumerate(r_values)]          
        
        
        
 
    def start(self):
        ###########################################################
        #Called from start_record when the start button was pressed
        #Clears sData buffer
        #Starts mic and starts polling mic which calls mic_callback 
        ###########################################################
        
        self.empty_buffer()
        levels = []
        
        self.mic.start()
        print("Completed self.mic.start()")
        Clock.schedule_interval(self.readbuffer, 1/samples_per_second)
        
        print("samples per second")
        print(samples_per_second)
        print("def start(self)")
        a = datetime.now().strftime('%d-%m-%Y %H:%M:%S') 
        print(a)
 
    def readbuffer(self, dt):
        print("we at readbuffer now")
        self.mic.poll()
 
    def dummy(self, dt):
        print ("dummy")
        
        
    def empty_buffer(self):
        self.sData = []
        
        
        
    def stop_close_audio(self):
        ################################################################
        #This function is called from the stop button 
        ###############################################################   
    
    
        #Clock.schedule_once(self.dummy, 0.5)
        
        #Stop the microphone polling so no more buffer data coming in
        Clock.unschedule(self.readbuffer)
        #Stop the microphone
        self.mic.stop()  
        #Clear all buffers      
        self.sData = []
        self.copy_sData = []
        self.decoded = []
        self.decoded_copy = []
        
        
        
             
                
 
    def stop(self,dt):
        ################################################################
        #This function is called from mic_callback using a clock schedule_once
        #that is started when recording is finished after 2 seconds
        #This function creates the wave file and does the inference to match the 
        #audio from the model
        ###############################################################
    
    
    
        Clock.schedule_once(self.dummy, 0.1)
        
        '''
        #Stop the microphone polling so no more buffer data coming in
        Clock.unschedule(self.readbuffer)
        #Stop the microphone
        self.mic.stop()
        '''
        
        #Get a copy of the audio data to analyze it. Allow sData to continue collecting audio for the graph 
        self.copy_sData = self.sData
      
        
        #Start creating the wave file
        wf = wave.open(PATH, 'wb')
        wf.setnchannels(self.mic.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.mic.rate)
        wf.writeframes(b''.join(self.copy_sData))
        print("we at stop")
        wf.close()
        
        print("files in pwd")
        print(os.listdir())
        
        print("contents of sData")
        print(self.sData)
        print("contents of copy_sData")
        print(self.copy_sData)        
        

        
        
        print("finished creating wav file")     
        b = datetime.now().strftime('%d-%m-%Y %H:%M:%S') 
        print(b)
        #No need to play the file
        #self.play()                    
        self.audio_path  = "rec_test1.wav"   
             
        #Downsample the audio from the microphones 44100 down to 16000 that the model was trained on
        self.audio_path_out  = "rec_test2.wav"
        self.audio_inrate = 44100
        self.audio_outrate = 16000
        self.audio_inchannels = 1
        self.audio_outchannels = 1      
        
        #Start downsampling of wave file as the model is trained on 16000 and we receiving 44100
        try:
            self.downsample_success = self.downsampleWav(self.audio_path, self.audio_path_out, self.audio_inrate, self.audio_outrate, self.audio_inchannels, self.audio_outchannels) 
        except:
            print("self.downsample_success failed")    
            self.downsample_success = False   
                
        if self.downsample_success == True:
            g = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
            print("downsampling completed at this time")
            print(g) 
            try:                 
                self.prepare_audio_frames()
            except:
                print("self.prepare_audio_frames() failed")
        else:
            print("Could not downsample")          
        


    '''        
    def play(self):
        MediaPlayer = autoclass('android.media.MediaPlayer')
        AudioManager = autoclass('android.media.AudioManager')

        self.sound = MediaPlayer()
        #self.sound.setDataSource(yourDataSource) #you can provide any data source, if its on the devie then the file path, or its url if you are playing online
        #self.sound.setDataSource('testaudio.mp4') 
        #self.audio_path = self.storage_path + "/wav/output2.wav"
        #self.audio_path = self.storage_path + "/wav/output1.wav" ##cant find folder
     
        
        
        
        #self.audio_path = dirCheck1 + "/wav/output1.wav"
        #self.audio_path = "/storage/emulated/0/org.example.c4k_tflite_audio1/wav/output1.wav"##Not found
        #self.audio_path = "/data/data/org.example.c4k_tflite_audio1/files/app/wav/output2.wav"
        #self.audio_path = "/data/data/org.example.c4k_tflite_audio1/files/app/output2.wav"        
        #self.audio_path  = "/data/data/org.example.c4k_tflite_audio1/files/app/testaudio.mp4"
        self.audio_path  = "rec_test1.wav"
     
        self.sound.setDataSource(self.audio_path) 
        self.sound.prepare()
        self.sound.setLooping(False) #you can set it to true if you want to loop     
        
        
     
        print("start play")
        e = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("time of start playing audio")
        print(e)
        self.sound.start()
        
        # You can also use the following according to your needs
        #self.sound.pause()
        #self.sound.stop()
        #self.sound.release()
        #self.sound.getCurrentPosition()
        #self.sound.getDuration()   
        
        #self.downsample()
        #Downsample the audio from the microphones 44100 down to 16000 that the model was trained on
        self.audio_path_out  = "rec_test2.wav"
        self.audio_inrate = 44100
        self.audio_outrate = 16000
        self.audio_inchannels = 1
        self.audio_outchannels = 1
        
        print("end of play")
        f = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("end time of playing audio")
        print(f)        
        
    '''    
   
      
      
      
    def downsampleWav(self, src, dst, inrate, outrate, inchannels, outchannels):
        '''
        if not os.path.exists(src):
            print('Source not found!')
            return False

        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        '''
        
        h = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("downsampling started at this time")
        print(h)  
        
        try:
            s_read = wave.open(src, 'r')
            s_write = wave.open(dst, 'w')
        except:
            print('Failed to open files!')
            return False

        n_frames = s_read.getnframes()
        data = s_read.readframes(n_frames)

        try:
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            if outchannels == 1 & inchannels != 1:
                converted[0] = audioop.tomono(converted[0], 2, 1, 0)
        except:
            print('Failed to downsample wav')
            return False

        try:
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted[0])
        except:
            print('Failed to write wav')
            return False

        try:
            s_read.close()
            s_write.close()
        except:
            print('Failed to close wav files')
            return False

        return True      
      
       
      
        
 
  
    def prepare_audio_frames(self):
    
        ###################################################
        #Step 4 - Prepare wav file to be tested . Test different examples. When live on android use the captured sound to test with so you dont need 
        ######################################################
        
        self.random_audio = self.audio_path_out

        #self.random_audio = 'House Sparrow.wav'
        #self.random_audio = 'White-breasted Wood-Wren.wav'
        #self.random_audio = 'Azara_s Spinetail.wav'
        #self.random_audio = 'Chestnut-crowned Antpitta.wav'
        #self.random_audio = 'Red Crossbill.wav'  
        #self.random_audio = 'House_Sparrow_downsampled.wav' 
 
 
        self.wav_file = wave.open(self.random_audio, 'rb')
        # from .wav file to binary data in hexadecimal
        self.binary_data = self.wav_file.readframes(self.wav_file.getnframes())
        # from binary file to samples
        self.audio_data = np.array(struct.unpack('{n}h'.format(n=self.wav_file.getnframes()*self.wav_file.getnchannels()), self.binary_data))
        #print("output from wave")
        #print(audio_data)
        #print(len(audio_data))

        print("audio before tf.int16.max")
        print(self.audio_data)
        
        
        
        '''
        
        Fs = 8000
        n_fft = 256
        frequency1 = np.arange(0, 1 + n_fft / 2) * Fs / n_fft
        print("frequency1")
        print(frequency1)
        
        
        #signal2 = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
        #signal2 = self.audio_data    might have to do to float
        signal2 = self.audio_data
        fourier = np.fft.fft(signal2)
        n = signal2.size
        timestep = 0.1
        freq2 = np.fft.fftfreq(n, d=timestep)
        print("frequency2")
        print(freq2)       
        '''
        
        
        # Load the example clip
        y, sr = librosa.load(librosa.ex('nutcracker'))

        # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        hop_length = 512

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Beat track on the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

        # And the first-order differences (delta features)
        mfcc_delta = librosa.feature.delta(mfcc)

        #  Stack and synchronize between beat events
        # This time, we'll use the mean value (default) instead of median
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        beat_chroma = librosa.util.sync(chromagram,beat_frames,aggregate=np.median)

        # Finally, stack all beat-synchronous features together
        beat_features = np.vstack([beat_chroma, beat_mfcc_delta])     
        print("beat")
        print(beat_features)   
        
        
        
        
        

        #tf.int16.max prints out as 32767 so just use this instead
        #audio_data = np.array(audio_data) / tf.int16.max

        self.audio_data = np.array(self.audio_data) / 32767

        #input_size = serving_model.input_shape[1]

        ###This is the size of the frame. 0.975 secomds long and is 16000 samples = 15600. try changing but you need
        #to train the model on the different rate as well
        #Dont need input size not using tf
        #input_size = 15600
        #print("input size")
        #print(input_size)
        print("audio before tf.signal.frame")
        print(self.audio_data)


        #def framing(sig, fs=16000, win_len=0.025, win_hop=0.01): ###This is for a spectogram
        #def framing(sig, fs=15600, win_len=0.025, win_hop=0.01):
        #def framing(sig, fs=15600, win_len=1, win_hop=0.975):
        #def framing(sig, fs=15600, win_len=1, win_hop=0.01):  ####runs forever
        def framing(sig, fs=15600, win_len=1, win_hop=1):   #####working one
            """
            transform a signal into a series of overlapping frames.

            Args:
            sig            (array) : a mono audio signal (Nx1) from which to compute features.
            fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
            win_len        (float) : window length in sec.
                                 Default is 0.025.
            win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.

            Returns:
            array of frames.
            frame length.
            """
            # compute frame length and frame step (convert from seconds to samples)
            frame_length = win_len * fs
            frame_step = win_hop * fs
            signal_length = len(sig)
            frames_overlap = frame_length - frame_step

            # Make sure that we have at least 1 frame+
            num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

            # Pad Signal to make sure that all frames have equal number of samples
            # without truncating any samples from the original signal
            if rest_samples != 0:
                pad_signal_length = int(frame_step - rest_samples)
                z = np.zeros((pad_signal_length))
                pad_signal = np.append(sig, z)
                num_frames += 1
            else:
                pad_signal = sig

            # make sure to use integers as indices
            frame_length = int(frame_length)
            frame_step = int(frame_step)
            num_frames = int(num_frames)

            # compute indices
            idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
            idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
            indices = idx1 + idx2
            self.frames = pad_signal[indices.astype(np.int32, copy=False)]
            return self.frames



        #Split the received audio into segments of 15600 frames and pad the end with 0's
        #splitted_audio_data = tf.signal.frame(audio_data, input_size, input_size, pad_end=True, pad_value=0)

        #Use the numpy version not tensorflow
        self.splitted_audio_data = framing(self.audio_data)


        #waveform = splitted_audio_data[0]
        #waveform = np.float32(waveform)
        self.waveform = self.splitted_audio_data
        self.waveform = np.float32(self.waveform)
        print("waveform data")
        print(self.waveform)

        #This will show how many rows of frames there are eg (28, 15600) is 28 rows of frames
        print("waveform.shape")
        print(self.waveform.shape) 
 
        try:
            self.start_inference()
        except:
            print("self.start_inference() failed")





 
 
    def start_inference(self):    
       
        ######################################################################
        #Step 6 - Start inference
        #####################################################################
        global audio_button_displayed_count
        ##
        
        
        '''
        #Do a fake graph plot now as we have stopped the mic input else the graph stops plotting
        ##
        try:
            #call plot graph
            self.get_value(self.queue_frames)      
            print("we have plotted the fake graph")
            print("fake self.queue_frames")
            print(self.queue_frames)      

        except:           

            pass
        '''
        
        
        
        results_array = []
        for i, data in enumerate(self.waveform):
            print("data in enumerate")
            print(data)
            #waveform = np.reshape(waveform, (1, 15600))
            try:
                inference_results = self.perform_inference(self.waveform_input_index,data)
                results_array.append(inference_results)
                print("results in enumerate at inference")
                print(results_array)
            except:
                print("self.perform_inference(self.waveform_input_index,data) failed")    

  
  


        results_np_array = np.array(results_array)
        print("results_np_array")
        print(results_np_array)
        mean_results_array = results_np_array.mean(axis=0)
        print("mean_results_array at inference")
        print(mean_results_array)
        result_index_array = mean_results_array.argmax()
        print("result_index_array")
        print(result_index_array)
        
        #Print the highest value at that index
        print("Highest value for result_index_array")
        self.highest_inference_value = mean_results_array[0,result_index_array]
        
        
        self.highest_inference_value = float(self.highest_inference_value )
        self.highest_inference_value = round(self.highest_inference_value,2)
        print(self.highest_inference_value)
        
        
        #print(f'Mean result: {test_data.index_to_label[result_index]} -> {mean_results[result_index]}')
        #print(_label_list_1[top_class_index]) 
        print("bird class")
        print(self._label_list[result_index_array])  
        self.bird_class = self._label_list[result_index_array]
        
        j = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("End of prediction")
        print(j)  
        
        
        #Now we need to display the info button for the bird.
        #****************************************************#
        #Get the info from the csv file using the birdname - note add birds from audio model
        


        file = open("csv/object_info.csv", "r")
        data = list(csv.reader(file, delimiter=","))
        file.close() 
        print("file data")   
        print(data)  
        
        index_of_list_value = next(i for i,v in enumerate(data) if self.bird_class in v)
        print('index_of_list_value')
        print(index_of_list_value)
        
        #list_row = data[[3][0]]
        list_row = data[[index_of_list_value][0]]
        print('list_row = data[[index_of_list_value][0]]')
        print(list_row)
        
        #self.detected_object_name_1 = list_row[1]
        #print('sef.detected_object_name_1')
        #print(self.detected_object_name_1)
        
        
        
        
       
        self.detected_object_name = list_row[0]
        self.object_general_info = list_row[1]
        self.object_info_1 = list_row[2]
        self.object_info_2 = list_row[3]
        print('self.detected_object_name')
        print(self.detected_object_name)
        print('self.object_general_info')
        print(self.object_general_info)
        print('self.object_info_1')
        print(self.object_info_1)
        print('self.object_info_2')
        print(self.object_info_1)
        
        
        
        
        ###########################################################
        #Only display the button if the inference value is high ie a good match in the model
        ##########################################################
        if(self.highest_inference_value > 0.58):              
        
            self.display_audio_detection_button = ButtonLayoutAudioDetection()
        
            
            '''
            self.display_audio_detection_button_value = 0
        
            if(self.display_audio_detection_button_value == 0):   
                #self.layout.add_widget(self.start_object_detection_button) 
                self.add_widget(self.display_audio_detection_button) 
                self.display_audio_detection_button_value = 1
            
            #Keep track of how many buttons are displayed
            audio_button_displayed_count = audio_button_displayed_count + 1   
            '''
            
            
            
            
            #Layer the buttons from bottom to top on the screen by using the position hint
            #Divide the audio_button_displayed_count by 10 to get 0.1, 0.2,0.3,0.4
            #self.position_hint_x = 0.1
            self.position_hint_x = 0.5
            #self.position_hint_y = audio_button_displayed_count/10  
            self.position_hint_y = 0.5 
        
            ########################################################
            #Display the info button showing the birds name
            #######################################################

            if(self.display_audio_detection_button_active == True):
                #If the button is displayed already then remove it to show next bird
                self.remove_widget(self.display_audio_detection_button)
                
                
            self.add_widget(self.display_audio_detection_button)  
            #Remove the button after 5 seconds
            Clock.schedule_once(self.remove_audio_detection_button,5)
            
              
            
            self.display_audio_detection_button.add_btn(self.bird_class,self.position_hint_x,self.position_hint_y,self.object_general_info,self.object_info_1,self.object_info_2)          
            #Activate button
            self.display_audio_detection_button_active = True
            
            
            '''
            if(audio_button_displayed_count == 5):
               audio_button_displayed_count = 0
               #Remove the previous buttons 
            '''   
               
        
        #######################################################################################
        #Note that self.recording_has_started is now False. We reset it from True after the scores so as to prevent any new recording starting before we finished analysing
        #self.recording_has_started = False will now activate the recording if the amplitude is high enough
        ########################################################################################3
        self.recording_has_started = False 
        #Clear sData
        self.empty_buffer()
        #Clear the copy of sData
        self.copy_sData = []
        #self.start()
 



    def remove_audio_detection_button(self,dt):   
        self.remove_widget(self.display_audio_detection_button)
        self.display_audio_detection_button_active = False
    
    
    

    def perform_inference(self,wave_input_index,audio_frame):

        ######################################################################
        #Step 5 - loop through audio frames and perform inference
        #####################################################################

        #setup as a method 
        audio_frame = np.reshape(audio_frame, (1, 15600))
        self.interpreter_audio.allocate_tensors()
        self.interpreter_audio.set_tensor(wave_input_index, audio_frame)
        print("waveform_input_index")
        print(wave_input_index)
        self.interpreter_audio.invoke()
        #The below one is to use the default model - dont use for now
        #scores = interpreter.get_tensor(scores_output_index)
        #scores = interpreter.get_tensor(test_data.index_to_label)
        #scores = interpreter.get_tensor(index_to_label)
        #scores = interpreter.get_tensor(test_data)
        #Below is the correct one to use for the bird model
        #self.scores = self.interpreter_audio.get_tensor(self.scores_output_index1)
        self.scores, self.embeddings, self.spectrogram = self.interpreter_audio.get_tensor(self.scores_output_index1)
    
        top_class_index = self.scores.argmax()
        print("top_class_index")
        print(top_class_index)

        print(self._label_list[top_class_index])  # Should print code for bird.
        #print(_label_list[spec_result_index])  # Should print name for bird.
        #print("scores")
        #print(scores)
        #print(scores.shape)  # Should print (1, 5)            
         
        #Allow the next recording to start      
         

        
        return self.scores
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            

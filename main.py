#from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.clock import Clock
from applayout import MDScreen
from applayout import AppLayout
from applayout import ButtonsLayout
from applayout import ButtonsLayoutInfo
from applayout import ButtonLayoutObjectDetection
from applayout import Content
from applayout import ContentAudio
from applayout import Settings
from applayout import Weather
from applayout import RecordForm
#from applayout import Example
from android_permissions import AndroidPermissions
from kivymd.app import MDApp
from kivy.uix.button import Button
#from applayout import ClassifyObject
#import pandas as pd
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
import csv
import json
import requests
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivy.lang import Builder

 
 
#if not os.path.isdir("/sdcard/kivyrecords/"):
#    os.mkdir("/sdcard/kivyrecords/")




# You can create your kv code in the Python file
Builder.load_string("""




#:import utils kivy.utils
<ScreenOne>:
    ###Home
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            #pos: self.pos
            size: self.size   


    #padding: 4
    #size_hint: None, None
    #size: "200dp", "100dp"

    MDFloatLayout:
        md_bg_color: 1,1,1,1
      
        
        MDIconButton:
            #icon: "icons/magnify-scan.png"
            icon: "camera-outline"

            #increment_width: "164dp"
            #width: dp(280)
            #user_font_size: "64sp"
        
 
            theme_text_color: "Custom"
            text_color: app.theme_cls.primary_color
            pos_hint: {"center_x": .2, "center_y": .8}
  
            on_press: 
                #root.manager.transition = NoTransition()                
                root.manager.current = 'app_layout'   
                app.start_object_detection_screen()             
        


        MDLabel: 
            id: location           
            text: "Verify Bird"
            pos_hint: {"center_x": .2, "center_y": .7}
            halign: "center"
            font_size: "20sp"
            
            
            
        MDIconButton:
            #icon: "icons/magnify-scan.png"
            icon: "volume-high"

            #increment_width: "164dp"
            #width: dp(280)
            #user_font_size: "64sp"
        
 
            theme_text_color: "Custom"
            text_color: app.theme_cls.primary_color
            pos_hint: {"center_x": .7, "center_y": .8}
  
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'record_form'   
                app.start_recorder_screen() 
   
            
        MDLabel: 
            id: temperature           
            text: "Audio ID"
            markup: True
            pos_hint: {"center_x": .7, "center_y": .7}
            halign: "center"
            font_size: "20sp"    
            



            

            
        MDIconButton:
            #icon: "icons/magnify-scan.png"
            icon: "information-outline"

            #increment_width: "164dp"
            #width: dp(280)
            #user_font_size: "64sp"
        
 
            theme_text_color: "Custom"
            text_color: app.theme_cls.primary_color
            pos_hint: {"center_x": .2, "center_y": .6}
  
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'screen_three'               
        


        MDLabel: 
            id: about          
            text: "About"
            pos_hint: {"center_x": .2, "center_y": .5}
            halign: "center"
            font_size: "20sp"
            
            
            
        MDIconButton:
            #icon: "icons/magnify-scan.png"
            icon: "bird"

            #increment_width: "164dp"
            #width: dp(280)
            #user_font_size: "64sp"
        
 
            theme_text_color: "Custom"
            text_color: app.theme_cls.primary_color
            pos_hint: {"center_x": .7, "center_y": .6}
  
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'screen_four'    
   
            
        MDLabel: 
            id: Species           
            text: "Bird Species"
            markup: True
            pos_hint: {"center_x": .7, "center_y": .5}
            halign: "center"
            font_size: "20sp"               

       

    

  
<ScreenTwo>:
    ####Weather
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            #pos: self.pos
            size: self.size   


    #padding: 4
    #size_hint: None, None
    #size: "200dp", "100dp"


    MDFloatLayout:
        md_bg_color: 1,1,1,1
        Image:
            source: "icons/map-marker.png"
            size_hint: .1,.1
            pos_hint: {"center_x": .5, "center_y": .95}

        MDLabel: 
            id: location           
            text: "East London, EL"
            pos_hint: {"center_x": .5, "center_y": .89}
            halign: "center"
            font_size: "20sp"
            
        Image:
            id: weather_image
            source: "icons/sun.png"
            size_hint: .1,.1
            pos_hint: {"center_x": .5, "center_y": .77}            
            
        MDLabel: 
            id: temperature           
            text: "[b]40[/b]°"
            markup: True
            pos_hint: {"center_x": .5, "center_y": .62}
            halign: "center"
            font_size: "60sp"    
            
            
        MDLabel: 
            id: weather          
            text: "Partly Cloudy"       
            pos_hint: {"center_x": .5, "center_y": .54}
            halign: "center"
            font_size: "20sp"   
            
            
            
            
                  

                                         

                
        MDFloatLayout:   
            pos_hint: {"center_x": .25, "center_y": .3}    
            size_hint: .22,.1
                 
            Image:             
                source: "icons/hightide.png"           
                pos_hint: {"center_x": .1, "center_y": .4}            
            
            MDLabel:    
                id:high1                     
                text: "4:33 AM SAST 0.8 m"             
                pos_hint: {"center_x": 1, "center_y": .6}            
                font_size: "18sp"                            
            
            MDLabel:                         
                text: "High 1"             
                pos_hint: {"center_x": 1, "center_y": .2}            
                font_size: "14sp"    
                                   
                      
        MDFloatLayout:   
            pos_hint: {"center_x": .7, "center_y": .3}    
            size_hint: .22,.1
                 
            Image:             
                source: "icons/waves.png"           
                pos_hint: {"center_x": .1, "center_y": .4}            
            
            MDLabel:    
                id:low1                   
                text: "12:40 PM SAST 0.7 m"             
                pos_hint: {"center_x": 1.1, "center_y": .6}            
                font_size: "16sp"                            
            
            MDLabel:                         
                text: "Low 1"             
                pos_hint: {"center_x": 1.1, "center_y": .2}            
                font_size: "14sp"    
                                        
                                         
        MDFloatLayout:   
            pos_hint: {"center_x": .25, "center_y": .2}    
            size_hint: .22,.1
                 
            Image:             
                source: "icons/hightide.png"           
                pos_hint: {"center_x": .1, "center_y": .3}            
            
            MDLabel:    
                id:high2                     
                text: "4:33 AM SAST 0.8 m"             
                pos_hint: {"center_x": 1, "center_y": .5}            
                font_size: "18sp"                            
            
            MDLabel:                         
                text: "High 2"             
                pos_hint: {"center_x": 1, "center_y": .1}            
                font_size: "14sp"    
                                   
                      
        MDFloatLayout:   
            pos_hint: {"center_x": .7, "center_y": .2}    
            size_hint: .22,.1
                 
            Image:             
                source: "icons/waves.png"           
                pos_hint: {"center_x": .1, "center_y": .3}            
            
            MDLabel:    
                id:low2                   
                text: "12:40 PM SAST 0.7 m"             
                pos_hint: {"center_x": 1.1, "center_y": .5}            
                font_size: "16sp"                            
            
            MDLabel:                         
                text: "Low 2"             
                pos_hint: {"center_x": 1.1, "center_y": .1}            
                font_size: "14sp"    
                                                    
                


        MDFlatButton:
            md_bg_color: app.theme_cls.primary_color
            #pos_hint: {"top": 1, "right": 1}
            pos_hint: {"center_x": .25, "center_y": .1}
            text: "check"
            on_press: app.get_weather("Dehli")
        
        
        
        MDFlatButton:
            md_bg_color: app.theme_cls.primary_color
            #pos_hint: {"top": 1, "right": 1}
            pos_hint: {"center_x": .7, "center_y": .1}
            text: "back"
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'screen_one'    
            
            
            
            
            
 
<ScreenThree>:
    ####About
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            #pos: self.pos
            size: self.size        



    #padding: 4
    #size_hint: None, None
    #size: "200dp", "100dp"

   

    MDRelativeLayout:
    

        MDLabel:
            id: label
            text: "About"
            adaptive_size: True
            color: "grey"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .9}
            bold: True
                
    
    
        MDLabel:
            id: label
            text: "FishID SA is an app that identifies South African angling fish."
            adaptive_size: True
            color: "black"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .8}
            bold: False        

            
        MDLabel:
            id: label
            text: ""
            adaptive_size: True
            color: "grey"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .4}
            bold: True


        MDFlatButton:
            md_bg_color: app.theme_cls.primary_color
            #pos_hint: {"top": 1, "right": 1}
            pos_hint: {"center_x": .5, "center_y": .4}
            text: "back"
            #on_press: app.remove_settings()
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'screen_one'    
 
<ScreenFour>:

    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            #pos: self.pos
            size: self.size        



    #padding: 4
    #size_hint: None, None
    #size: "200dp", "100dp"

   

    MDRelativeLayout:
    

        MDLabel:
            id: species
            text: "Fish Species"
            adaptive_size: True
            color: "grey"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .9}
            bold: True
                
    
    
        MDLabel:
            id: label
            text: "Fish species in South Africa."
            adaptive_size: True
            color: "black"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .8}
            bold: False        

            
        MDLabel:
            id: label
            text: ""
            adaptive_size: True
            color: "grey"
            #pos: "12dp", "12dp"
            pos_hint: {"center_x": .5, "center_y": .4}
            bold: True


        MDFlatButton:
            md_bg_color: app.theme_cls.primary_color
            #pos_hint: {"top": 1, "right": 1}
            pos_hint: {"center_x": .5, "center_y": .4}
            text: "back"
            on_press: 
                #root.manager.transition = NoTransition()
                root.manager.current = 'screen_one'    


 
<ScreenFive>:
    BoxLayout:
        Button:
            text: "Go to Screen 1"
            background_color : 1, 0, 0, 1
            on_press:
                #root.manager.transition.direction = 'right'
                #root.manager.transition = NoTransition()
                root.manager.current = 'app_layout'
                

                                
 
 
""")

class ScreenOne(MDScreen):
    pass
  
class ScreenTwo(MDScreen):
    pass
 
class ScreenThree(MDScreen):
    pass
 
class ScreenFour(MDScreen):
    pass
 
class ScreenFive(MDScreen):
    pass
    

class ScreenManager(ScreenManager):
    pass
    #current = "screen_one"            
  
  







#3Builder.load_file('setupkv.kv')
#Builder.load_file('setupkv_1.kv')

kv = Builder.load_file('setupkv.kv')
#kv = Builder.load_file('setupkv_1.kv')

if platform == 'android':
    from jnius import autoclass
    from android.runnable import run_on_ui_thread
    from android import mActivity
    View = autoclass('android.view.View')

    @run_on_ui_thread
    def hide_landscape_status_bar(instance, width, height):
        # width,height gives false layout events, on pinch/spread 
        # so use Window.width and Window.height
        if Window.width > Window.height: 
            # Hide status bar
            option = View.SYSTEM_UI_FLAG_FULLSCREEN
        else:
            # Show status bar 
            option = View.SYSTEM_UI_FLAG_VISIBLE
        mActivity.getWindow().getDecorView().setSystemUiVisibility(option)
elif platform != 'ios':    
    # Dispose of that nasty red dot, required for gestures4kivy.
    from kivy.config import Config 
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

 


class MyApp(MDApp):

    dialog = None
    
    
    
    def build(self):
        #Initialize continue detection
        print("We at def build(self)")
        self.continue_detection_value = 0  
        #Initialize info button
        self.info_button_value = 0         
        
        self.started = False
        if platform == 'android':
            Window.bind(on_resize=hide_landscape_status_bar)


        # The ScreenManager controls moving between screens
        self.screen_manager = ScreenManager(transition=NoTransition())
        #self.screen_manager = ScreenManager(transition=NoTransition())




        
        
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        
        '''
        self.layout = MDScreen()        
        self.cameradetect = AppLayout()    
        self.layout.add_widget(self.cameradetect)   
        self.addinfobutton = ButtonsLayoutInfo()  
        '''
       
       
       

        
        
        '''
        #Add button to start object detection
        self.start_object_detection_button = ButtonLayoutObjectDetection()
        
        self.start_object_detection_button_value = 0
        
        if(self.start_object_detection_button_value == 0):   
            self.layout.add_widget(self.start_object_detection_button) 
            self.start_object_detection_button_value = 1
            
        self.start_detection = 'Start'
        self.start_object_detection_button.add_btn(self.start_detection )        
        '''
        
        
        
       
       
        #Add RecordForm class widget
        self.recorder = MDScreen(name="record_form")     
        self.layout = MDScreen(name="app_layout")
           
        
        self.addinfobutton = ButtonsLayoutInfo()  
        
        
        
             
        #self.screen_manager.add_widget(self.layout(name ="app_layout"))
        self.screen_manager.add_widget(ScreenOne(name ="screen_one"))
        self.screen_manager.add_widget(ScreenTwo(name ="screen_two"))
        self.screen_manager.add_widget(ScreenThree(name ="screen_three"))
        self.screen_manager.add_widget(ScreenFour(name ="screen_four"))
        self.screen_manager.add_widget(ScreenFive(name ="screen_five"))       
        #self.screen_manager.add_widget(AppLayout(name ="app_layout"))
        
        
        self.screen_manager.add_widget(self.layout)
        self.screen_manager.add_widget(self.recorder)
        
        
        #To remove widget - nb clock still runs camera in background
        #self.layout.remove_widget(self.cameradetect)         
        
           
        #self.addcamerabutton = ButtonsLayout()
        #self.layout.add_widget(self.addcamerabutton) 
       
        
        #self.layout.add_widget(self.addinfobutton)
  
        
      
         
        #return self.layout
        return self.screen_manager
       
        
        
        
        #return kv

    def start_object_detection_screen(self):                 
        self.cameradetect = AppLayout()         
        self.layout.add_widget(self.cameradetect) 
        print("Added AppLayout class widget")        
        
      
 
        
        
    def start_recorder_screen(self):                 
        self.recorderform = RecordForm()          
        self.recorder.add_widget(self.recorderform) 
        print("Added RecordForm class widget")    
          
               
        
    def navigation_draw(self):
        print("Navigation")    

    def on_start(self, *args):
        self.dont_gc = AndroidPermissions(self.start_app)

    def start_app(self):
        self.dont_gc = None
       
        # Can't connect camera till after on_start()
        
        #Clock.schedule_once(self.on_stop, 10)
        #Clock.schedule_once(self.on_start, 20)
        
        
        ######For testing audio disconnect the camera detect for now
        #*******************************************************
        #Clock.schedule_once(self.connect_camera)
        
        
    def start_object_detection(self):
        #Called from start button on camera view
        Clock.schedule_once(self.connect_camera)      
        
        

    def connect_camera(self,dt):    
        #self.layout.detect.connect_camera(enable_analyze_pixels = True)
        #MDApp.root.applayout.detect.connect_camera(enable_analyze_pixels = True)
        #self.root.layout.detect.connect_camera(enable_analyze_pixels = True)
        self.cameradetect.detect.connect_camera(enable_analyze_pixels = True)
        
      
        #pass
        
        
    def screenshot(self):
        self.cameradetect.detect.capture_screenshot()
        #self.parent.addcamerabuttondetect.detect.capture_screenshot()
        #self.cameradetect.detect.connect_camera(enable_analyze_pixels = True)
        #self.detect.capture_screenshot()
        #self.addcamerabutton.detect.capture_screenshot()
        #self.parent.addcamerabutton.detect.capture_screenshot()
        #self.addcamerabutton.detect.capture_screenshot()
        #self.applayout.detect.capture_screenshot()       
        print("Did screenshot") 
       

    def on_stop(self, *args):
        
        #self.layout.detect.disconnect_camera()
        #self.layout.detect.disconnect_camera()
        self.cameradetect.detect.disconnect_camera()
        del self.cameradetect
        print("camera stopped")
        
        
    def remove_camera_layout(self, *args):
    
        
        
        #self.layout.detect.disconnect_camera()
        #self.layout.detect.disconnect_camera()
        #self.cameradetect.detect.disconnect_camera()
        self.layout.remove_widget(self.cameradetect)
        self.layout.remove_widget(self.addcamerabutton)
        print("remove_camera_layout")
        #self.layout.add_widget(self.cameradetect) 
        #print(MyApp.get_running_app())
        
  
    def add_camera_layout(self, *args):
        
        #self.layout.detect.disconnect_camera()
        #self.layout.detect.disconnect_camera()
        #self.cameradetect.detect.disconnect_camera()
        #self.layout.remove_widget(self.cameradetect)        
        self.layout.add_widget(self.cameradetect)       
        self.layout.add_widget(self.addcamerabutton) 
    
        print("added_camera_layout")      
        
        
        
        
              

    def add_object_info_layout(self, *args):
        
        #self.layout.detect.disconnect_camera()
        #self.layout.detect.disconnect_camera()
        #self.cameradetect.detect.disconnect_camera()
        #self.layout.remove_widget(self.cameradetect)        
        #self.layout.add_widget(self.cameradetect)  
        
        lst = []
            
        #self.layout.remove_widget(self.addinfobutton) 
        print("added_object_info_layout")  
        print("image name") 
        #for arg in args:
            #self.detected_object_name = arg
            #print(self.detected_object_name)
            #self.new_text = self.detected_object_name
            
            
        for arg in args:
            #self.new_text = arg
            lst.append(arg)
            #print("args")    

        self.detected_object_name = lst[0]
        self.probability_highest_value = lst[1]    
            
            
            
            
        


        file = open("csv/object_info.csv", "r")
        data = list(csv.reader(file, delimiter=","))
        file.close() 
        print("file data")   
        print(data)  
        
        index_of_list_value = next(i for i,v in enumerate(data) if self.detected_object_name in v)
        print('index_of_list_value')
        print(index_of_list_value)
        
        #list_row = data[[3][0]]
        list_row = data[[index_of_list_value][0]]
        
        #self.detected_object_name_1 = list_row[1]
        #print('sef.detected_object_name_1')
        #print(self.detected_object_name_1)
        
        
        
        
       
        self.detected_object_name = list_row[0]
        self.object_general_info = list_row[1]
        self.object_info_1 = list_row[2]
        self.object_info_2 = list_row[3]
        
        
        
        #self.object_all_info = self.detected_object_name +"\n" + self.object_general_info + "\n" + self.object_info_1
        self.object_all_info = self.object_general_info
            
        #self.addinfobutton = ButtonsLayoutInfo() 
        
        #Only add info button once. Thereafter move it out of sight and back in sight again
        print("self.info_button_value")
        print(self.info_button_value)
        if(self.info_button_value == 0):   
            self.layout.add_widget(self.addinfobutton) 
            self.info_button_value = 1
            
        
        #self.layout.remove_widget(self.addinfobutton) 
        #self.addinfobutton.remove_btn(self.addinfobutton)
       
        #self.addinfobutton.add_btn(self.detected_object_name, self.object_general_info, self.object_info_1)
        self.addinfobutton.add_btn(self.detected_object_name, self.object_all_info, self.probability_highest_value)
        
        #Set a timer that removes the info button after x amount of seconds if the user does
        #not open the dialog box in time 
        self.remove_button_time_expired(self)
        
        
        #self.continue_detection_value set value to 1 to stop detecting
        self.continue_detection_value = 1
        
    def remove_button_time_expired(self, *args):  
        print("we are going to remove the button as the timer will expire")     
        self.time_expired = Clock.schedule_once(self.remove_button_time_expired_confirmed,5)
   


    def cancel_remove_button_time_expired(self, *args):  
        #We need to cancel the clock schedule to remove the info button as the user
        #clicked on the Dialog box in time
        print("The dialog box was opened so stop the remove button cloack schedule")     
        self.time_expired.cancel()
        
        


       


    def remove_button_time_expired_confirmed(self, *args):  
        print("we have removed the button as the timer has expired")     
        Clock.schedule_once(self.addinfobutton.remove_btn,1)
        self.continue_detection_value = 0

        
    '''    
    def show_alert_dialog(self):
            #Stop the clock schedule from removing the info button as dialog was opened
            self.cancel_remove_button_time_expired(self)
            
            if not self.dialog:
                self.dialog = MDDialog(
                    #size_hint= [0.9, None],
                    size_hint= [0.8, 0.5],
                    #Dont allow the user to close the dialog by click outside the box
                    auto_dismiss = False,
                    #title=self.detected_object_name,
                    #text=self.object_all_info,                   
                    type="custom",
                    #content_cls=Content(self),
                    content_cls=Content(),
                    buttons=[
                      
                        MDFlatButton(
                            text="Close",
                            on_release=self.closeDialog,
                            theme_text_color="Custom",
                            text_color=self.theme_cls.primary_color,
                        ),
                    ],
                )
            #Prevent the user from closing the dialog box by clicking outside the area    
            #self.dialog.bind(on_dismiss=self.dont_close_dialog)    
            self.dialog.open() 
            self.dialog.content_cls.add_img_name(self.detected_object_name, self.object_all_info, self.object_info_1, self.object_info_2)  
             
            #self.remove_info_button_layout(self)
               

    #def dont_close_dialog(self,instance):
        #print('Popup', instance, 'is being dismissed but is prevented!')
        #return True   
    
    
    def closeDialog(self, inst):
        self.dialog.dismiss()
        print("closeDialog pressed")
        
        #Continue to detect images in classifyobject.py        
        #self.continue_detection_value set value to 0 to continue detecting
        self.continue_detection_value = 0
       
            
        #self.layout.remove_widget(self.addinfobutton) 
        #self.addinfobutton.remove_btn(self.addinfobutton)
        
    '''    
        

    def continue_detection(self, *args):   
            continue_to_detect = self.continue_detection_value
      
            return continue_to_detect
       
 
 

    def show_alert_dialog(self):   
        print("we at show_alert_dialog")
        #Stop the clock schedule from removing the info button as dialog was opened
        self.cancel_remove_button_time_expired(self)
       
        self.adddialog = Content()  
      
        self.layout.add_widget(self.adddialog)   
        self.adddialog.add_img_name(self.detected_object_name, self.object_all_info, self.object_info_1, self.object_info_2)   
        
        
        


    def remove_show_alert_dialog(self, *args):   
        print("we removed show_alert_dialog")
        #self.addsettings = Settings()  
        #self.layout.add_widget(self.addsettings) 
        self.layout.remove_widget(self.adddialog)  
        #self.remove_widget(self.adddialog)
     
        #Continue to detect images in classifyobject.py        
        #self.continue_detection_value set value to 0 to continue detecting
        self.continue_detection_value = 0
        
        




    def show_alert_dialog_audio(self,detected_object_name_audio,object_all_info, object_info_1, object_info_2):   
        print("we at show_alert_dialog_audio")
        print(detected_object_name_audio)
        #Stop the clock schedule from removing the info button as dialog was opened
        #self.cancel_remove_button_time_expired(self)
       
        self.adddialog_audio = ContentAudio()  
      
        #self.layout.add_widget(self.adddialog)   
        self.recorder.add_widget(self.adddialog_audio)
       
        self.adddialog_audio.add_img_name(detected_object_name_audio, object_all_info, object_info_1, object_info_2)      
     
        
        #self.adddialog.add_img_name(self.detected_object_name, self.object_all_info) 

        
    #def remove_dialog(self, *args): 
    def remove_dialog_audio(self):      
        print("we at def remove_dialog_audio")
        #self.addsettings = Settings()  
        #self.layout.add_widget(self.addsettings) 
        #self.layout.remove_widget(self.adddialog)  
        self.recorder.remove_widget(self.adddialog_audio)
     
        #Continue to detect images in classifyobject.py        
        #self.continue_detection_value set value to 0 to continue detecting
        self.continue_detection_value = 0      


    def remove_record_layout(self): 

        print("def remove_record_layout")
        self.recorder.remove_widget(self.recorderform)  
        del self.recorderform
        self.screen_manager.current = 'screen_one'



    def remove_applayout_layout(self): 

        print("remove_applayout_layout")     
        self.layout.remove_widget(self.cameradetect)  
        
        self.screen_manager.current = 'screen_one'














    def settings(self, *args):   
        print("we at settings")
       
        self.addsettings = Settings()  
        self.layout.add_widget(self.addsettings)    


    def remove_settings(self, *args):   
        print("we removed settings")
        #self.addsettings = Settings()  
        #self.layout.add_widget(self.addsettings) 
        self.layout.remove_widget(self.addsettings)   
        


    def weather(self, *args):   
        print("we at weather")
       
        self.addweather = Weather()  
        self.layout.add_widget(self.addweather)    


    def remove_weather(self, *args):   
        print("we removed weather")
        #self.addsettings = Settings()  
        #self.layout.add_widget(self.addsettings) 
        self.layout.remove_widget(self.addweather)  
        
        
        
    def get_weather(self, city_name):   
        print("we at get weather")
        try:
        
       
            #url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}$appid={self.api_key}"
            #response = requests.get(url)
            
                     
            json_data_weather = '{"location": "East London", "temp": 282.21,"humidity": 65,"cod": "200","id": 2.55,"wind_speed": 20,"weather": "Clear sky"}'
            json_data_tides = '{"Day": "Wed 01", "High1": "5:32 AM SAST 0.94 m","Low1": "11:44 AM SAST 1.26 m","High2": "6:27 PM SAST 0.83 m","Low2":"7:22 PM SAST 0.74 m"}'          
                        
                
   
                 
            #x = response.json()     
            #x = json.dumps(json_data)
            x = json.loads(json_data_weather)
            y = json.loads(json_data_tides)
                
            
            print(x)
            print(type(x))
            
            #if x["cod"][0] != "404":
            a = 0
            if(a == 0):
                temperature = round(x["temp"]-273.15)
                humidity = x["humidity"]
                wind_speed = round(x["wind_speed"]*18/5)
                weather = x["weather"]
                location = x["location"]
                
                self.addweather.ids.temperature.text = f"[b]{temperature}[/b]°"
                self.addweather.ids.weather.text = str(weather)
                self.addweather.ids.humidity.text = f"{humidity}%"
                self.addweather.ids.wind_speed.text = f"{wind_speed} km/h"
                self.addweather.ids.location.text = str(location)


                high1 = y["High1"]
                low1 = y["Low1"]
                high2 = y["High2"]
                low2 = y["Low2"]                
      
                
               
                self.addweather.ids.high1.text = str(high1)
                self.addweather.ids.low1.text = str(low1)
                self.addweather.ids.high2.text = str(high2)
                self.addweather.ids.low2.text = str(low2)


                
                
            else:
                print("City not found")   
              
                 
        except requests.ConnectionError:
            print("No internet conection")
        
        
    def search_weather(self):   
        print("we at search weather")
        self.get_weather("Delhi")
                       


MyApp().run()


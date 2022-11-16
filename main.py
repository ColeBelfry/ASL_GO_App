#app layout imports
from kivy.app import App

#kivy UX components
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
#other kivy components
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import NumericProperty
#other dependencies
import os
import cv2
import tensorflow as tf
import numpy as np
import json
import time
#Paths to imported models (.pb files)
WORDS_MODEL_PATH = './models/asl_words_mobnet/saved_model'
ALPHABET_MODEL_PATH = './models/asl_alphabet_mobnet/saved_model'
ICONS_PATH = './icons'
WORDS_MODEL_INDEX_PATH = './models/asl_words_mobnet/category_index_dict.json'
ALPHABET_MODEL_INDEX_PATH = './models/asl_alphabet_mobnet/category_index_dict.json'

#Screen Manager
sm = ScreenManager()


#Application screens
'''
Landing Screen: the home screen that directs the user to one of the detection model screens.
Loading Screen: the screen displayed between screen transitions where there is a wait time
Alphabet Tutorial Screen: the screen displayed before the alphabet model to help the user use the model more effectively
Alphabet Tutorial Screen: the screen displayed before the words model to help the user use the model more effectively
Alphabet Detect Screen: the screen displayed to detect the alphabet
Words Screen: the screen displayed to detect words'''
################################################################
class landing_screen(Screen):
    def __init__(self, **kwargs):
        super(landing_screen, self).__init__(**kwargs)
        self.has_visited_alphabet = False
        self.has_visited_words = False

    def start(self):
        pass

    def on_click_alphabet(self, *args):
        if(self.has_visited_alphabet == True):
            self.change_screen('loading_screen')
            Clock.schedule_once(lambda dt: self.change_screen('alphabet_screen'), 10)
        else:
            self.has_visited_alphabet = True
            self.change_screen('alphabet_tutorial_screen')

    def on_click_words(self, *args):
        if(self.has_visited_words == True):
            self.change_screen('loading_screen')
            Clock.schedule_once(lambda dt: self.change_screen('words_screen'), 10)
        else:
            self.has_visited_words = True
            self.change_screen('words_tutorial_screen')

    def change_screen(self, screen_name):
        screen = self.manager.get_screen(screen_name)
        screen.start()
        self.manager.current = screen_name
#################################################################################
class loading_screen(Screen):
    def __init__(self, **kwargs):
        super(loading_screen, self).__init__(**kwargs)
        self.progress_val = 1

    def start(self):
        self.ids.loadcircle.start()
    def leave(self):
        self.ids.loadcircle.stop()
##########################################################################################

class alphabet_tutorial_screen(Screen):
    def __init__(self, **kwargs):
        super(alphabet_tutorial_screen, self).__init__(**kwargs)
    def start(self):
        pass
    def leave(self):
        pass
    def on_click_continue(self, *args):
        self.change_screen('loading_screen')
        Clock.schedule_once(lambda dt: self.change_screen('alphabet_screen'), 10)
    def on_click_back(self, *args):
        self.change_screen('landing_screen')
    def change_screen(self, screen_name):
        screen = self.manager.get_screen(screen_name)
        screen.start()
        self.manager.current = screen_name
##########################################################################################

class words_tutorial_screen(Screen):
    def __init__(self, **kwargs):
        super(words_tutorial_screen, self).__init__(**kwargs)
    def start(self):
        pass
    def leave(self):
        pass
    def on_click_continue(self, *args):
        self.change_screen('loading_screen')
        Clock.schedule_once(lambda dt: self.change_screen('words_screen'), 10)
    def on_click_back(self, *args):
        self.change_screen('landing_screen')
    def change_screen(self, screen_name):
        screen = self.manager.get_screen(screen_name)
        screen.start()
        self.manager.current = screen_name
##########################################################################################
class alphabet_detect_screen(Screen):
    def __init__(self, **kwargs):
        super(alphabet_detect_screen, self).__init__(**kwargs)
        #Variables
        self.detect = False
        self.capture = None
        self.event = None
        self.time = None

        start_time = time.time()
        self.model = tf.saved_model.load(ALPHABET_MODEL_PATH)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(' Alphabet Model Loaded. Took {} seconds.'.format(elapsed_time))

        self.catagory_index_list = self.get_cat_index()
        self.min_score_threshold = .75
        self.label_id_offset = 1
        self.translation_out_threshold = 30
        self.translation_list = []

    def start(self):
        self.detect = True
        self.capture = cv2.VideoCapture(0)
        self.event = Clock.schedule_interval(self.update, 1.0/33.0)
        

    #Must run a countinous update function for video capture and detection
    def update(self, *args):
        #Capture
        ret, frame = self.capture.read()
        if(self.detect):
            #Preprocessing
            image_np = np.array(frame)
            image_np = np.expand_dims(image_np, 0)
            #image_np = np.resize(image_np, (1,320,320,3))
            input_tensor = tf.convert_to_tensor(
                image_np, dtype=tf.uint8)

            #Detect
            detections = self.model(input_tensor)
            
            #Output
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            #detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)

            #Getting the detection classes above our minimum score threshold and putting them to a list
            temp_list = []
            for i in range(num_detections):
                detection_score = detections['detection_scores'][i]
                if(detection_score > self.min_score_threshold):
                    class_id = detections['detection_classes'][i]
                    class_name = self.catagory_index_list[class_id - 1]['name']
                    #Special case to handle O favor
                    if(class_name == 'O' and detection_score < .95):
                        detections['detection_scores'][i] = .6
                    else:
                        temp_list.append(class_name)
            self.translation_list.extend(temp_list)

        #Formatting output image for display. This requires a texture
        frame = frame[120:120+Window.size[0], 120:120+Window.size[0], :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.abfeed.texture = img_texture

        #Logic to get the most prominent detection
        if(len(self.translation_list) >= self.translation_out_threshold):
            most_frequent_class = max(set(self.translation_list), key = self.translation_list.count)
            print(self.translation_list)
            print('name: ' + most_frequent_class)
            self.update_translation_box(most_frequent_class)
            self.translation_list.clear()
            #self.time = time.time()
        #cur_time = time.time()
        #if(self.time != None and cur_time - self.time > 10):
        #    print(self.translation_list)
        #    self.translation_list.clear()

    def on_click_home(self, *args):
        self.detect = False
        self.ids.pausebtn.text = "Pause"
        self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)
        self.event.cancel()
        self.capture.release()
        self.translation_list.clear() 
        self.manager.current = 'landing_screen'

    def on_click_words(self, *args):
        self.detect = False
        self.ids.pausebtn.text = "Pause"
        self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)
        self.event.cancel()
        self.capture.release()
        self.translation_list.clear()
        self.change_screen('loading_screen')
        Clock.schedule_once(lambda dt: self.change_screen('words_screen'), 10)

    def on_click_pause(self, *args):
        if self.detect == True:
            self.detect = False
            self.translation_list.clear()
            self.ids.pausebtn.text = "Unpause"
            self.ids.pausebtn.background_color = (46/255,134/255,171/255,1)
        elif self.detect == False:
            self.detect = True
            self.ids.pausebtn.text = "Pause"
            self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)
    def on_click_clear(self, *args):
        self.ids.abtboxout.text = ""
        self.translation_list.clear()

    def change_screen(self, screen_name):
        screen = self.manager.get_screen(screen_name)
        screen.start()
        self.manager.current = screen_name
    
    def get_cat_index(self):
        with open(ALPHABET_MODEL_INDEX_PATH) as json_file:
            data_arr = json.load(json_file)
            print('alphabet catagory_index loaded from json file')
            return data_arr
    #prints the detection string to the translation box on screen
    def update_translation_box(self, class_name):
        if(self.ids.abtboxout.text == ''):
            self.ids.abtboxout.text = class_name
        else:
            self.ids.abtboxout.text += (' ' + class_name)
####################################################################

class word_detect_screen(Screen):
    def __init__(self, **kwargs):
        super(word_detect_screen, self).__init__(**kwargs)
        #Variables
        self.detect = False
        self.capture = None
        self.event = None
        self.time = None

        start_time = time.time()
        self.model = tf.saved_model.load(WORDS_MODEL_PATH)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(' Words Model Loaded. Took {} seconds.'.format(elapsed_time))
        
        self.catagory_index_list = self.get_cat_index()
        self.min_score_threshold = .8
        self.label_id_offset = 1
        self.translation_out_threshold = 35
        self.translation_list = []

    def start(self):
        self.detect = True
        self.capture = cv2.VideoCapture(0)
        self.event = Clock.schedule_interval(self.update, 1.0/33.0)

    #Must run a countinous update function for video capture and detection
    def update(self, *args):
        #Capture
        ret, frame = self.capture.read()
        
        if(self.detect):
            image_np = np.array(frame)
            image_np = np.expand_dims(image_np, 0)
            #image_np = np.resize(image_np, (1,320,320,3))
            input_tensor = tf.convert_to_tensor(
                image_np, dtype=tf.uint8)

            #Detect
            detections = self.model(input_tensor)

            #Output
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)

            #Getting the detection classes above our minimum score threshold and putting them to a list
            temp_list = []
            for i in range(num_detections):
                detection_score = detections['detection_scores'][i]
                if(detection_score > self.min_score_threshold):
                    class_id = detections['detection_classes'][i]
                    class_name = self.catagory_index_list[class_id - 1]['name']
                    temp_list.append(class_name)
            self.translation_list.extend(temp_list)

        #Formatting output image for display. This requires a texture
        frame = frame[120:120+Window.size[0], 200:200+Window.size[0], :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.wfeed.texture = img_texture

        #Logic to get the most prominent detection
        if(len(self.translation_list) >= self.translation_out_threshold):
            most_frequent_class = max(set(self.translation_list), key = self.translation_list.count)
            print(self.translation_list)
            print('name: ' + most_frequent_class)
            self.translation_list.clear()
            self.update_translation_box(most_frequent_class)
            self.time = time.time()
        elif(self.time != None and time.time() - self.time > 6):
            self.translation_list.clear()
            

    def on_click_home(self, *args):
        self.detect = False
        self.ids.pausebtn.text = "Pause"
        self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)
        self.event.cancel()
        self.capture.release()
        self.translation_list.clear() 
        self.manager.current = 'landing_screen'

    def on_click_alphabet(self, *args):
        self.detect = False
        self.ids.pausebtn.text = "Pause"
        self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)
        self.event.cancel()
        self.capture.release()
        self.translation_list.clear()
        self.change_screen('loading_screen')
        Clock.schedule_once(lambda dt: self.change_screen('alphabet_screen'), 10)

    def on_click_pause(self, *args):
        if self.detect == True:
            self.detect = False
            self.translation_list.clear()
            self.ids.pausebtn.text = "Unpause"
            self.ids.pausebtn.background_color = (46/255,134/255,171/255,1)
        elif self.detect == False:
            self.detect = True
            self.ids.pausebtn.text = "Pause"
            self.ids.pausebtn.background_color = (73/255,201/255,255/255,1)

    def on_click_clear(self, *args):
        self.ids.wtboxout.text = ""
        self.translation_list.clear()

    def change_screen(self, screen_name):
        screen = self.manager.get_screen(screen_name)
        screen.start()
        self.manager.current = screen_name

    def get_cat_index(self):
        with open(WORDS_MODEL_INDEX_PATH) as json_file:
            data_arr = json.load(json_file)
            print('words catagory_index loaded from json file')
            return data_arr

    #formats the detection output to fit english grammar rules if applicable and prints it to the translation box
    def update_translation_box(self, class_name):
        break_out_flag = False
        if(self.ids.wtboxout.text == ''):
            self.ids.wtboxout.text = class_name
        else:
            txt_arr = self.ids.wtboxout.text.split(' ')
            last_word = txt_arr[len(txt_arr) - 1]
            out_str = ''
            if(last_word != class_name):
                #sorry
                if(break_out_flag == False and class_name == 'sorry'):
                    if(last_word == 'I'):
                        out_str += (' am ' + class_name)
                        break_out_flag = True
                #apple
                if(break_out_flag == False and class_name == 'apple'):
                    if(last_word == 'me'):
                        out_str += (' an ' + class_name)
                        break_out_flag = True
                    if(last_word == 'get'):
                        out_str += (' an ' + class_name)
                        break_out_flag = True
                    if(last_word == 'have'):
                        out_str += (' an ' + class_name)
                        break_out_flag = True
                    if(last_word == 'love'):
                        out_str += (' apples')
                        break_out_flag = True
                #good
                if(break_out_flag == False and class_name == 'good'):
                    if(last_word == 'I'):
                        out_str += (' am ' + class_name)
                        break_out_flag = True
                    if(last_word == 'you'):
                        out_str += (' are ' + class_name)
                        break_out_flag = True
                    if(last_word == 'apple'):
                        out_str += (' is ' + class_name)
                        break_out_flag = True
                #no
                if(break_out_flag == False and class_name == 'no'):
                    if(last_word == 'I'):
                        out_str += (' do not ')
                        break_out_flag = True
                    if(last_word == 'you'):
                        out_str += (' do not ')
                        break_out_flag = True
                    if(last_word == 'my'):
                        out_str += (' am not ')
                    if(last_word == 'can'):
                        out_str += (' not ')
                        break_out_flag = True
                #I
                if(break_out_flag == False and class_name == 'I'):
                    if(last_word == 'how'):
                        out_str += (' do ' + class_name)
                        break_out_flag = True
                    if(last_word == 'help'):
                        out_str += (' me ')
                        break_out_flag = True
                    if(last_word == 'like'):
                        out_str += (' me ')
                        break_out_flag = True
                    if(last_word == 'want'):
                        out_str += (' me ')
                        break_out_flag = True
                    if(last_word == 'get'):
                        out_str += (' me ')
                        break_out_flag = True
                #you
                if(break_out_flag == False and class_name == 'you'):
                    if(last_word == 'how'):
                        out_str += (' are ' + class_name)
                        break_out_flag = True
                #catch
                if(break_out_flag == False):
                    self.ids.wtboxout.text += (' ' + class_name)
                else:
                    self.ids.wtboxout.text += (out_str)
###########################################################################
class LoadCircleLbl(Label):

    angle = NumericProperty(0)
    update_clock = None
    stop_flag = False

    def __init__(self, **kwargs):
        super(LoadCircleLbl, self).__init__(**kwargs)
    def start(self):
        self.angle = 0
        self.update_clock = Clock.schedule_interval(lambda dt: self.set_Circle(1/60), 1/30)
    def stop(self):
        self.update_clock.cancel()
    def set_Circle(self, dt):
        self.angle = self.angle + dt*360
        if self.angle >= 360:
            self.angle = 0
###################################################################
class NavIconBtn(ButtonBehavior, Image):
    pass
class NavLabelBtn(ButtonBehavior, Label):
    pass
###################################################################
#Kivy Application Class
class ASLGOApp(App):

    def build(self):
        self.title = 'ASL GO Application'
        Window.size = (350, 622)
        # Build collection of our screens w/ the manager
        sm.transition = SlideTransition()
        sm.add_widget(loading_screen(name='loading_screen'))
        return sm

    def on_start(self):
        #if in debug mode
        os.chdir('./ASLGO_App')
        sm.add_widget(landing_screen(name='landing_screen'))
        sm.add_widget(alphabet_tutorial_screen(name='alphabet_tutorial_screen'))
        sm.add_widget(words_tutorial_screen(name='words_tutorial_screen'))
        load_screen = sm.get_screen('loading_screen')
        load_screen.start()
        #this changes the screen to the landing_screen
        Clock.schedule_once(lambda dt: self.change_screen(load_screen), 10)

    def change_screen(self, load_screen, *args):
        sm.add_widget(alphabet_detect_screen(name='alphabet_screen'))
        sm.add_widget(word_detect_screen(name='words_screen'))
        #progress_update.cancel()
        sm.current = 'landing_screen'
        load_screen.leave()
        print('changing screen')

    
if __name__ == '__main__':
    ASLGOApp().run()

import os
import matplotlib as matplot
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import math
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

csv_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "csv")

class Animation :
    def __init__(self, filename) :
        self.data_path = os.path.join(csv_dir, filename)
        
    def csv_to_frames(self) :
        '''
        Takes the traffic dataframe and turns each frame into a .jpg picture
        '''
        df = read_data(self.data_path)
        maxFrameNum = int(max(df['Frame #']))    # Find the maximum number of frame
        xmin, xmax = 200, 1000
        ymin, ymax = 0, 100
        aspectRatio = (xmax-xmin) / (5 * (ymax - ymin))
        
        for i in range(730, 770):
            # Plot dimension setup
            fig, ax = plt.subplots(figsize=(15,8))
            ax.set_aspect(aspectRatio)
            plt.xlim(xmin, xmax)
            plt.ylim(0, ymax)
            plt.xlabel('longitude')
            plt.ylabel('lateral')

            plt.figure(i+1)

            # extract the ID & road coordinates of the bottom 4 points of all vehicles at frame # i
            frameSnap = df.loc[(df['Frame #'] == i)]
            frameSnap = np.array(frameSnap[['lateral','longitude','length','width','ID','class']])

            # Looping thru every car in the frame
            for j in range(len(frameSnap)):  
                carID = frameSnap[j,4]
                carClass = frameSnap[j,5]
                
                # Road Coordinates of the Car
                top_right_x = frameSnap[j,1]
                top_right_y = frameSnap[j,0] + (frameSnap[j,3] / 2)
                bottom_right_x = top_right_x
                bottom_right_y = frameSnap[j,0] - (frameSnap[j,3] / 2)
                top_left_x = top_right_x - frameSnap[j,2]
                top_left_y = top_right_y
                bottom_left_x = top_left_x
                bottom_left_y = bottom_right_y
                
                coord = np.array([top_right_x, top_right_y, top_left_x, top_left_y,\
                                 bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y])
                coord = np.reshape(coord,(-1,2)).tolist()
                coord.append(coord[0])
                xs, ys = zip(*coord)

                # Displaying information above the car
                plt.text(top_right_x, top_right_y, int(carID), fontsize='large')

                # Plotting the car
                oneCarColor = getCarColor(carClass)
                ax.plot(xs, ys, c = oneCarColor)
                ax.fill(xs, ys, c = oneCarColor)

            plt.title(i, fontdict={'fontsize':'x-large','fontweight':'bold'}, pad=20)
            plt.savefig('../Animation/Pictures/' + format(i,"04d") + '.jpg', dpi=100)
            

    def animate(self) :
        image_folder = '../Animation/Pictures/US101'
        video_name = '../Animation/Videos/US101.mp4'

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort()

        frame = cv2.imread(os.path.join(image_folder, images[1]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 10, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

        
def read_data(data_path) :
    df = pd.read_csv(data_path)
    df = df[['Vehicle_ID','Frame_ID','Local_X','Local_Y','v_length','v_Width','v_Class',\
             'Lane_ID','Preceding','Following','Space_Headway']]
    df.columns = ['ID','Frame #','lateral','longitude','length','width','class',\
                  'Lane','Preceding','Following','Space_Headway']
    return df


def getCarColor(carClass) :
    if(carClass == 1) : return 'red'
    elif(carClass == 2) : return 'yellow'
    elif(carClass == 3) : return 'green'
    else : return 'black'
    
    
    
    
    
    
    
    
    
    
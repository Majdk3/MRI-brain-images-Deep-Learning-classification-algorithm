
import final_results_2
import predict_seg_one_image
import predict_class

import numpy as np
import cv2
import os
def play(dirc):


    orginal_image=cv2.imread(dirc)
    orginal_image=cv2.resize(orginal_image,(512,512))


    from predict_class import predct
    predicted_text,x=predct(dirc)
    
    
    from predict_seg_one_image import predict_seegmented
    segmented_image=predict_seegmented(dirc)

    

    
    
    
    final_results_2.final_results(orginal_image,segmented_image,predicted_text)
    

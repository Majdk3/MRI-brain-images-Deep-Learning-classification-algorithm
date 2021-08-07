

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def final_results(orginal_image,segmented_image,predicted_text):
    fig3 = plt.figure(constrained_layout=True)
    grid = fig3.add_gridspec(2, 3)

    ax1_predicted=plt.subplot(grid[1,1])
    ax1_segmented=plt.subplot(grid[0,1])
    ax1_image=plt.subplot(grid[0,0])
    ax1=plt.subplot(grid[0,2])
    
    ax1.set_title("mask applied on orginal image")
    
    ax1_segmented.set_title("Segmented tumor mask")  
    ax1_image.set_title("Orginal image  ")  

 
    cmap = plt.cm.Greens
    red = cmap(np.arange(cmap.N))
    red[:, -1] = np.linspace(0, 1, cmap.N)
    red = ListedColormap(red)

    cmap = plt.cm.gray
    white = cmap(np.arange(cmap.N))
    white[:, -1] = np.linspace(0, 1, cmap.N)
    white = ListedColormap(white)
    
    
   
    
    ax1.imshow(orginal_image[:,:,0],
               cmap='gray', interpolation='none')
    
  
    ax1.imshow(segmented_image[:,:,0], cmap=red,
               alpha=0.6, interpolation='none')

    ax1.axis('off')
    ax1.grid(b=None)
    ax1.set_title('mask applied')



    segmented_image=segmented_image[:,:,0]
    orginal_image=orginal_image[:,:,0]
    
    ax1_segmented.imshow(segmented_image,cmap="gray")
    ax1_image.imshow(orginal_image,cmap="gray")
   



    ax1_predicted.axis([0 ,512 ,0 ,512])
    ax1_predicted.axis('off')
    
    ax1_predicted.text(53, 247, 'Predicted tumor type : \n     {}'.format(predicted_text), style='italic',fontsize=32,
        bbox={'facecolor': 'red', 'alpha': 0.5})
    fig3.canvas.manager.full_screen_toggle() #  fullscreen mode
    fig3.show()
    fig3.set_size_inches((8.5, 11), forward=False)

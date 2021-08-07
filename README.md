# MRI-brain-images-Deep-Learning-classification-algorithm

# Abstract
Classification and segmentation of brain tumors in medical images is an important step when diagnosing, monitoring, and treating the disease. This is usually done manually by a radiologist, but recent years have seen an increase in the use of neural networks to classify and segment tumors in MRI images of patients. In this work we use two deep neural networks (DNNs), a U-Net and a CNN, to replicate results achieved by other researchers using a similar approach.
Two Deep Learning models based on convolutional neural networks and U-Nets are proposed to classify and segment different brain tumor types respectively. Using a publicly available MRI image dataset which contains 3 types of tumors (meningiomas, gliomas, and pituitary tumors). The dataset includes 233 patients with a total of 3064 images on T1-weighted contrast-enhanced images. The proposed network structures achieved a significant performance with the best overall accuracy of 96.35% for classification and 71.7% for segmentation. The results indicate the ability of the models for assistive brain tumor diagnosis purposes.

# N.B:
the result models are too big to be uploaded here, so here's a gdrive link to them:
https://drive.google.com/drive/folders/1W_Mn1pwovOzNYUO5yFXzY0kbfWLQr4dx?usp=sharing

# The used Dataset:
The brain tumor dataset contains 3064 T1-weighted contrast-inhanced images
from 233 patients with three kinds of brain tumor: meningioma (708 slices), 
glioma (1426 slices), and pituitary tumor (930 slices).The 5-fold
cross-validation indices are also provided.

This data is organized in matlab data format (.mat file). Each file stores a struct
containing the following fields for an image:
cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
cjdata.PID: patient ID
cjdata.image: image data
cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.
		For example, [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on tumor border.
		It was generated by manually delineating the tumor border. So we can use it to generate
		binary image of tumor mask.
cjdata.tumorMask: a binary image with 1s indicating tumor region

The data was then exported as png files using the mfile script, which of course as I later realized is quite redundant, as you can load the .mat dataset directly into Python, . at the time i started this project this was however still unknown to me and due to time constraints i never got around to it, but thought i should at least mention it.

Also note that data augmentation is used (Zoom, horizontal flip, and Shear) in order to increase the data size as it's subpar for deep learning use in its current state.

Here's a link to the dataset and you'll find below an example from it:
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679


![image](https://user-images.githubusercontent.com/88331345/128345850-88accf21-0bcf-494f-a865-8ffd64584054.png)


# Classification:

# Deephealth
## DeepHealth - project is created in Project Oriented Deep Learning Training program. The program is organized by Deep Learning.
## The repository includes:


- Source code of U-net
- Instruction and training code for the BraTS dataset.
- Pre-trained weights on MS COCO.
- notebooks to visualize the detection result for Mask RCNN.
- # Introduction
The aim of this project is to distinguish gliomas which are the most difficult brain tumors to be detected with deep learning algorithms. Because, for a skilled radiologist, analysis of multimodal MRI scans can take up to 20 minutes and therefore, making this process automatic is obviously useful.

MRI can show different tissue contrasts through different pulse sequences, making it an adaptable and widely used imaging technique for visualizing regions of interest in the human brain. Gliomas are the most commonly found tumors having irregular shape and ambiguous boundaries, making them one of the hardest tumors to detect. Detection of brain tumor using a segmentation approach is critical in cases, where survival of a subject depends on an accurate and timely clinical diagnosis.

We present a fully automatic deep learning approach for brain tumor segmentation in multi-contrast magnetic resonance image.

# Requirements
- Numpy
- Scipy
- Pillow
- Cython
- Matplotlib
- Scikit-image
- Tensorflow>=1.3.0
- Keras>=2.0.8
- OpenCV-Python
- h5py
- imgaug
- IPython[all]

## 1. Dataset 
dataset can be downloade from open source kaggle

## 2. Build datset 
dataset stored in two foldere "image", "mask" and then divided into train and validation directories

## 3. Build Model 

We use only one U-net model to do three different segmentation tasks Full Tumor, Tumor Core, Enhancing Tumor.![u-net](https://user-images.githubusercontent.com/85225054/168588179-1c5d7a69-af09-4d11-b10f-5431033d764f.png)

## Result![output](https://user-images.githubusercontent.com/85225054/168588384-9cfb05c1-8f36-421a-a8a7-f4300c569173.png)

## References



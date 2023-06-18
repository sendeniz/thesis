# Sequence Models for Video Object Detection
 
**General:**
<br>
This repo contains avarious base torch implementation of sequence models, such as a 1) Simple RNN, GRU RNN, LSTM RNN, HIPPO RNN and Transformer. They can either be used independently or embedded within a YoloV4 object detection, turning the still object detector into a video object detecting, since the sequence models are able to capture the spatio-temporal signal. A short demo of our detection system can be seen in Fig. 1. The full demonstration can be found [here](https://www.youtube.com/watch?v=Q30_ScFp8us). 

<p align="center">
  <img src="figures/yolov1_demo.gif" alt="animated" />
  <figcaption>Fig.1 - Real-time inference using YoloV1. </figcaption>
</p>


**Example Dictionary Structure**

<details>
<summary style="font-size:14px">View dictionary structure</summary>
<p>

```
.
├── application                		# Real time inference tools
    └── __init__.py 
    └── yolov4_watches_you.py  		# YoloV4 inference on webcam 
    └── yolov4_watches_youtube.py	# YoloV4 inference on an .mp4 video file in `video/`
├── cpts				# Weights as checkpoint .cpt files
    └── efficentnet_yolov4_mscoco.cpt	# Pretrained yolov4 still-image detector
    └── efficentnet_yolov4_imagenetvid.cpt	# Pretrained yolov4 video detector
├── figures                    		# Figures and graphs
    └── ....
├── loss                       		# Custom PyTorch loss
    └── __init__.py  		
    └── yolov3_loss.py
├── models                     		# Pytorch models
    └── __init__.py  		
    └── darknet.py
    └── yolov1net_darknet.py		# Original YoloV1 backbone (not supported: no backbone weights available)
    └── tiny_yolov1net.py		# Original tiny YoloV1 backbone 
    └── tiny_yolov1net_mobilenetv3_large.py # Mobilenetv3 size M/large backbone
    └── tiny_yolov1net_mobilenetv3_small.py # Mobilenetv3 size S/small backbone
    └── tiny_yolov1net_squeezenet.py 	# Squeezenet backbone 
    └── yolov1net_resnet18.py		# Resnet18 pre-trained backbone
    └── yolov1net_resnet50.py		# Resnet50 pre-trained backbone
    └── yolov1net_vgg19bn.py		# Vgg19 with batchnormalization pre-trained backbone
├── results                    		# Result textfiles
    └── ....
├── train                      		# Training files
    └── __init__.py  
    └── train_darknet.py
    └── train_yolov1.py 
├── utils                      		# Tools and utilities
    └── __init__.py
    └── custom_transform.py		# Image transformation/augmentation
    └── darknet_utils.py		
    └── dataset.py
    └── figs.py.			# Create figures
    └── generate_csv.py			# Create training and testing csv files
    └── get_data.sh			# Fetch data and assign into appropriate folder structure
    └── get_data_macos.sh		
    └── get_inference_speed.py		# Get inference speed
    └── iou_map_tester.py		# mAP tester
    └── voc_label.py			
    └── yolov1_utils.py			
├── video                      		
    └── youtube_video.mp4		# .mp4 video from youtube
    └── yolov1_watches_youtube.mp4      # Result of `yolov1_watches_youtube.py`
├── requierments.txt           		# Python libraries
├── setup.py                   		
├── terminal.ipynb             		# If you want to run experiments on google collab
├── LICENSE
└── README.md
```

</p></details>

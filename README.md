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
├── application                # Real time inference tools
    └── __init__.py 
    └── yolo_watches_you.py  		# Yolo inference on webcam or video you choose 
├── cpts				# Weights as checkpoint .cpt files
    └── ...
    └── efficentnet_yolov4_mscoco.cpt	# Pretrained yolov4 still-image detector
    └── efficentnet_yolov4_imagenetvid.cpt	# Pretrained yolov4 video detector
├── figures                    # Figures and graphs
    └── ....
├── loss                       # Custom PyTorch loss
    └── __init__.py  		
    └── yolov3_loss.py
├── models                     # Pytorch models
    └── __init__.py  		
    └── rnn.py                 # Rnns in base torch (simple, gru, lstm)
    └── yolov4.py		            # yolov4 architecture in base torch
├── results                    # Result textfiles
    └── ....
├── train                      # Training files
    └── __init__.py  
    └── train_rnn.py
    └── train_yolo.py 
├── utils                      	# Tools and utilities
    └── __init__.py
    └── graphs.py
    └── utils.py  			
├── requierments.txt           		# Python libraries
├── setup.py                   		
├── terminal.ipynb             		# If you want to run experiments from a notebook or on google collab
├── LICENSE
└── README.md
```

</p></details>

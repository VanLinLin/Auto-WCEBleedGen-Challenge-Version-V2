# Auto-WCEBleedGen-Challenge-Version-V2

## 1.Environment

Run the prepare.sh to auto create the virtual environment for classification and instance segmentation:

```bash
bash prepare_env.sh
```

### 1.1 Classification

Download the testing classification data from [here](https://drive.google.com/drive/folders/1MBT-x7fFPIWLCLSX0INQAqOtNL8J2CAa?usp=sharing).
Unzip the classification_data.zip and classification_weight.zip, moving them into classification folder.

```bash
YOUR_PATH\AUTO-WCEBLEEDGEN-CHALLENGE-VERSION-V2\CLASSIFICATION
├───config
│   ├───efficientnet
│   └───_base_
│       ├───datasets
│       │   └───pipelines
│       ├───models
│       └───schedules
├───data <- *From classificaiton_data.zip*
│   └───WCEBleedGen_v2
│       ├───test1
│       │   ├───bleeding
│       │   └───non-bleeding
│       ├───test2
│       │   ├───bleeding
│       │   └───non-bleeding
│       ├───train
│       │   ├───bleeding
│       │   └───non-bleeding
│       └───val
│           ├───bleeding
│           └───non-bleeding
├───datasets
│   └───pipelines
├───tools
└───weight <- *From classificaiton_weight.zip*
```

### 1.2 Instance segmentation
Download the testing instance segmentation data from [here](https://drive.google.com/drive/folders/1CYz6MMGqIOIKq9rU6c-PO_Fe3onjdCgB?usp=sharing).
Unzip the instance_segmentation_data.zip and instance_segmentation_weight.zip, moving them into instance_segmentation folder.
```bash
C:\USERS\USER\DESKTOP\AUTO-WCEBLEEDGEN-CHALLENGE-VERSION-V2\INSTANCE_SEGMENTATION
├───configs
│   ├───coco
│   └───_base_
│       ├───datasets
│       ├───models
│       └───schedules
├───data<- *From instance_segmentation_data.zip*
│   └───WCEBleedGen_v2
│       ├───instance_seg_img_test1
│       │   ├───Annotations
│       │   ├───coco_annotation
│       │   ├───Images
│       │   └───Labels
│       └───instance_seg_img_test2
│           ├───Annotations
│           ├───coco_annotation
│           ├───Images
│           └───Labels
├───mmcv_custom
│   └───__pycache__
├───mmdet_custom
│   ├───datasets
│   │   └───__pycache__
│   ├───models
│   │   ├───backbones
│   │   │   └───__pycache__
│   │   ├───dense_heads
│   │   │   └───__pycache__
│   │   ├───detectors
│   │   │   └───__pycache__
│   │   ├───utils
│   │   │   └───__pycache__
│   │   └───__pycache__
│   └───__pycache__
├───ops_dcnv3
│   ├───build
│   │   ├───bdist.linux-x86_64
│   │   ├───lib.linux-x86_64-3.9
│   │   │   ├───functions
│   │   │   └───modules
│   │   └───temp.linux-x86_64-3.9
│   │       └───ssd8
│   │           └───van
│   │               └───InternImage
│   │                   └───detection
│   │                       └───ops_dcnv3
│   │                           └───src
│   │                               ├───cpu
│   │                               └───cuda
│   ├───DCNv3.egg-info
│   ├───dist
│   ├───functions
│   │   └───__pycache__
│   ├───modules
│   │   └───__pycache__
│   └───src
│       ├───cpu
│       └───cuda
├───tools
└───weight <- *From instance_segmentation_weight.zip*
```
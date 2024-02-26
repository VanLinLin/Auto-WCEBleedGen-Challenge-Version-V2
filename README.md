# Auto-WCEBleedGen-Challenge-Version-V2

## 1.Environment

Run the prepare.sh to auto create the virtual environment for classification and instance segmentation:

```bash
bash prepare_env.sh
```

### 1.1 Classification

Download the testing data from [here](https://drive.google.com/drive/folders/1MBT-x7fFPIWLCLSX0INQAqOtNL8J2CAa?usp=sharing).
Unzip the classification_dat.zip and classification_weight.zip, moving them into classification folder.

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

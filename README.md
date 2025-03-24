# NYCU Computer Vision 2025 Spring HW1
- StudentID: 313553044
- Name: 江仲恩

## Introduction

## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW1.git
cd NYCU-Computer-Vision-2025-Spring-HW1
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW1
├── data
│   ├── test
│   ├── train
│   └── val
├── utils
│   ├── __init__.py
│   ├── early_stopping.py
│   ├── losses.py
│   .
│   .
│   .
├── environment.yml
├── main.py
├── train.py
├── test.py
.
.
.
```

4. Run for Train
    1. Train Model 
    ```
    python main.py DATAPATH [--epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--save SAVE_FOLDER] [--mode train]
    ```
    Example
    ```
    python main.py ./data --epochs 100 --batch_size 64 --learning_rate 5e-5 --decay 1e-4 --save saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```

## Performance snapshot
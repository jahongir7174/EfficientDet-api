# EfficientDet

EfficientDet implementation for object detection using Tensorflow2 Object detection API
### Usage
* Put your label file to the `nets/data` folder, see example in this folder
* Run `python generate.py` for generating train and test tensorflow record files. Please, change `set` and `output_path` arguments related to train and val set
* Run `python train.py` for training

### Dataset structure
    ├── Dataset folder 
        ├── IMAGES
            ├── 1111.jpg
            ├── 2222.jpg
        ├── LABELS
            ├── 1111.xml
            ├── 2222.xml
        ├── train.txt
        ├── val.txt
### Note 
* xml file should be PascalVOC format
* for making `train.txt` and `val.txt` format, see `VOC2012/ImageSets/Main/train.txt` 
### Reference
* https://github.com/tensorflow/models/tree/master/research/object_detection

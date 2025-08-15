Code associated with the paper, "Functional Ultrasound Imaging Combined with Machine Learning for Whole-Brain Analysis of Drug-Induced Hemodynamic Changes"

-- [SVM_validation_mk.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/SVM_validation_mk.py): Performs k-validation of a Support-Vector-Machine trained to classify MK/Saline mice. 

-- [dataset.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/dataset.py): Defines a PyTorch dataset class that flattens and labels MK-801 and saline fUSI data for GPU-ready model training and evaluation.

-- [model.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/model.py): Implements fixed and flexible PyTorch CNN architectures ending with global average pooling and a linear layer for binary classification and class activation map generation.

-- [normalize.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/normalize.py):Provides functions to normalize fUSI data by converting each pixel’s signal to percent change from a baseline period, for single or multiple mice.

-- [trainer.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/trainer.py): Implements training and evaluation functions for PyTorch models.

-- [validation_mk.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/validation_mk.py): Performs k-validation of a CNN trained to classify MK/Saline mice. 

-- [vit_rollout.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/vit_rollout.py): Generates normalized attention maps from Vision Transformer models using head fusion and attention rollout.

-- [vit_validation_mk.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/vit_validation_mk.py): Performs k-validation of a Vision Transformer trained to classify MK/Saline mice.

-- [viz.py](https://github.com/DeightonJared/fUSI_CAM/blob/main/viz.py): Provides functions to visualize fUSI data and class activation maps




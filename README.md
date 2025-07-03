### How to start a train script

Go to mtl_training to modify the following flags:

- denoised: True means you operate on images without medical annotations. This will just determine the dataset path
- cropped: True means you operate on the dataset of cropped images of only lesions.

After setting these flags in the code, you can run the train script:

python -m train_scripts.mtl_training --clinical --task joint --backbone efficientnet

Here are the required training flags:

- --clinical / --no-clinical : determine if clinical information (hospital, menopausal status) is used
- task: joint OR segmentation OR classification, joint means classification + semantic segmentation
- backbone: "efficientnet" is EfficientNetB0 and "classic" is a U-Net backbone

## Running inference with a model

After training a model, the test script uses the test dataset to evaluate the performance of this trained model. In order run the test script, run python -m test_scripts.test_hospital.

The test_scripts/test_hospital directory has the following 

## Note on paths
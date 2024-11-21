
Plate Detection - v2 V2
==============================

This dataset was exported via roboflow.com on November 20, 2024 at 2:56 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 607 images.
Plate are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -25 and +25 degrees
* Random shear of between -22° to +22° horizontally and -38° to +38° vertically
* Salt and pepper noise was applied to 1.84 percent of pixels



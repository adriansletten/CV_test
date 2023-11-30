# CV test
TASK: train an object detection with the supplied dataset and deploy it to the internet for inference. Model must be queried via POST requests. Optional task is to create a streamlit app for interactivety.



## Dataset
Dataset is available [here](https://universe.roboflow.com/adriansletten/lions_and_hippos).

The original dataset contains 3 classes: "lions", "hippopotamus" and "-". Each image contains 1 bounding box around one, one of, or several of the animals we are interested in. Below is some information about the images per class.
- **hippopotamus**: most of these images are quite large (more than 1000x1000). This class is the majority in the dataset
- **lion**: these images are of low resolution. Also quality of collected images is low: contains text, cartoon images, ...
- "**-**": class to be removed. Consists of 3 lion images with non-informative bounding boxes. (Can alternatively be reannotated)

In general the annotation quality (as in the bounding boxes) is not that good. Lacks consistency in regards to what should be annotated. Are we annotating all of the animals in the image, or are we just choosing one of the animals? For now this is unclear. Another problem with the annotations is that they sometimes just capture parts of the animals

For the next steps I first utilise the original dataset, then the dataset with the "-" class removed, and then finally the dataset using the mosaic augmentation (3x training set size). 


## Model training
For simplicity I utilise a yolo-v8 model. I train in google colab using [this notebook](model_training.ipynb). I load the dataset from the link in the previous section using the roboflow package, and using training/val/test scripts from the ultralytics package (distributors of yolo-v8). [This page](https://docs.ultralytics.com/usage/cfg/) contains parameters that can be changed when calling the different scripts.

### training setups
Before removing the "-" class, I trained with the default training setup for 100 epochs resizing the images to a size of 640x640. This led to strange and bad results.

Next I removed the "-" class and trained for 100 and 200 epochs. This resulted in good results for the "hippopotamus" class, but not so good for "lion". \
After testing the model on pictures from outside the dataset (random images found in google) a big flaw was discovered. The model had learned the correlation between class and resolution/bounding-box size. 

For the next training runs, attempts were made to reduce the models reliance on object size. I repeated training where the images were resized to 320x320 and 1024x1024. These did not help improve results significantly. I then discovered how this resizing is done: the largest image side is set to the chosen resolution applying padding where needed. Also I was discovered that small images are not upsampled. Related to these findings I also tried doing a square resizing of the images, but that did not help either. 

Following these failed attempts, I started looking into image augmentations that could help deal with the size issue. I settled with the mosaic augmentation, which extends the training set with mosaics/collages of 4 images. Mosaics (and the correct bounding boxes) are created so as to triple the training set size. That is I kept the original 200 images, and created 400 mosaics out of the 200 images. \
Using this new dataset I repeated training for 100 epochs and resizing to 640x640. The resulting model performs almost perfectly for both classes.


## Model deployment
The model is deployed directly to Roboflow at the end of the notebook. This model can then be queried using POST requests as described [here](https://docs.roboflow.com/deploy/hosted-api/object-detection#inference-api-parameters).

### Examples of usage
- [Simple POST query test](test_request.py)
- [Directly interact with the Roboflow Inference API](https://detect.roboflow.com)
- [Streamlit app](https://cvtest-lions-and-hippos.streamlit.app)

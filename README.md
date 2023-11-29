# CV test
TASK: train an object detection with the supplied dataset and deploy it to the internet for inference. Model must be queried via POST requests. Optional task is to create a streamlit app for interactivety.



## Dataset
Dataset is available [here](https://universe.roboflow.com/adriansletten/lions_and_hippos).

The original dataset contains 3 classes: "lions", "hippopotamus" and "-". Each image contains 1 bounding box around one, one of, or several of the animals we are interested in. Below is some information about the images per class.
- **hippopotamus**: most of these images are quite large (more than 1000x1000). This class is the majority in the dataset
- **lion**: these images are of low resolution. Also quality of collected images is low: contains text, cartoon images, ...
- "**-**": class to be removed. Consists of 3 lion images with non-informative bounding boxes. (Can alternatively be reannotated)

In general the annotation quality (as in the bounding boxes) is not that good. Lacks consistency in regards to what should be annotated. Are we annotating all of the animals in the image, or are we just choosing one of the animals? For now this is unclear. Another problem with the annotations is that they sometimes just capture parts of the animals

For the next steps we utilise the dataset with the "-" class removed.


## Model training
For simplicity we utilise a yolo-v8 model. We train in google colab using [this notebook](). We load the dataset from the link in the previous section using the roboflow package, and using training/val/test scripts from the ultralytics package (distributors of yolo-v8). [This page](https://docs.ultralytics.com/usage/cfg/) contains parameters that can be changed when calling the different scripts.

### training parameters
We train for 100 epochs resizing the images to a size of 640x640?. sweeps?....

### Model evaluation
add eval results? 


## Model deployment
The model is deployed directly to Roboflow at the end of the notebook. This model can then be queried using POST requests as described [here](https://docs.roboflow.com/deploy/hosted-api/object-detection#inference-api-parameters).


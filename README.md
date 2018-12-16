#  Deep Sketch Drawer -- CS230

*Authors: Yipeng He, Wantong Jiang, Di Bai*

This is our final project for CS230 Fall 2018 at Stanford.

There are two parts of our project.
First we developed a pipeline that converts a color image into an image that is in the style of sketch. We use HED network (https://github.com/senliuy/Keras_HED_with_model) to generate intermediate edge images and use the neural algorithm for art style (https://github.com/anishathalye/neural-style) to modify edge images and generate sketchs.

Second we tried other neural network approaches to generate sketches, which uses sketchy database data (http://sketchy.eye.gatech.edu/). Convolutional neural network pipelines are developed based on HED network mentioned above, and the generative adversarial model pipeline was generated based on a GAN structure used for quickdraw dataset (https://github.com/coreyauger/quickdraw-gan). We wrote a unified data parser for the two different pipelines.

we have also included our tensorflow code (under cnn&gan\tensorflow_cnn folder) used for milestone development
## Neural Style Transfer Based
### Quick Start

```
cd deep_sketch_drawer/neural-style-transfer
python neural_style.py
```
Enter the `neural-style-transfer` directory, and run `neural_style.py`.
It will automatically start generating sketches from `image_data/input_images/` and save sketches in `output/`.

#### Parameters
```

--input #change the input images's directory
--output # change the output sketches directory

```
Note that please check the directories you choose exist before running, also please make sure the required files said in `neural-style-transfer/model/` exist.

#### File Struture

- `image_data\` includes input colored images, intermediate edge images, and sketches from dataset used for training 'sketchy' style.
- `output\` will include generated sketches after running.
- `samples\` includes some sample edge images and generated sketches we produced.
- `model\` includes checkpoint.212-0.11.hdf5 and imagenet-vgg-verydeep-19.mat.If not, please download them before running.
- `produceHED.py` generates and saves edge images.
- `neural_style.py` reads and writes data, compute parameters for stylization.
- `stylize.py` trains and generates sketches.
- `hed.py` implements hed network.
- `vgg.py` implements vgg network.


## CNN&GAN based
### Quick start
```
cd deep_sketch_drawer/cnn-and-gan
python runScripts.py
```
Enter the `cnn-and-gan` directory and run   `runScripts.py`.
This will run the HED network algorithm and produce edge images

#### Parameters
```
--type #used to control which model to run, includes produceHED|hed|conv|gan|testCONV option
--cv_lr #used to pass learning rate for CNN models
--batch_size #used to control batch_size for models
--discrim_lr #used to control discriminator learning rate
--gan_lr #used to control learning rate for the generator&discriminator combined model
--epochs #used to control how many epochs to run
--testFile #used to provide model file for testing purpose
```

### Usage
To run the CNN & GAN based part of our project
Please create the following folders and structure them in the cnn&gan folder
- `colorFolder` contains the color image used for testing/HED detection
- `outputFolder` used to hold output images of the tested model / generated HED images
- `train_val\train\trainInput` contains input image for training
- `train_val\train\trainTarget` contains target image for training
- `train_val\val\valInput` contains input image for validation
- `train_val\val\valTarget` contains target image for validation
- `GANSketch` contains sketch image for GAN training
- `edgeSketch` contains edge image for GAN training

CNN models support loading data on the fly instead of loading all of them in memory at once.
You will need a file containing the training pairs and a file containing the validation pairs.
An example of trainng file has been provided in train_val.
Change the file names in the unified_data_loader.py or pass it as arguments to the DataParser class






# Model training

This is the code used to train the model used for 

## Usage

I used the dataset ["glasses or no glasses"](https://www.kaggle.com/aniruddha123/glasses-data). It contains roughly 5'000 images of people either wearing glasses or not wearing glasses. The dataset is roughly balanced, as roughly half of the images contain people not wearing glasses and vice versa.

## Training
download the dataset linked above and update the "dir" and "out_dir" variable to point to the dataset and the desired output directory respectively. It doesn't matter if you run the .ipynb or the .py program, as they contain equivalent code.

## The model
The model is an ensemble of three pre-trained models. In a first step the individual pre-trained models (Inception ResNet V2, XCeption and ResNet152 V2) get trained in a fashion usual in transfer learning. In a second step we concatenate the outputs of the second-to-last layer of these pretrained models and use two dense layers (with dropout layers to improve generalization) to get the final output of the model. Data augmentation (random cropping, contrast and rotation) gets employed during training to improve generalization. 

## The final model
the final trained model can be found [here](https://www.kaggle.com/yvesbischofberger/glassdetectionmodel). It achieves a validation accuracy of roughly 99% on the dataset.

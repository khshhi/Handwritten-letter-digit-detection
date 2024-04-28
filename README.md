# Handwritten Letter Recognition Model

## Overview:
This repository hosts code for training and evaluating a Convolutional Neural Network (CNN) model tailored for handwritten letter recognition. The model is trained on a hybrid dataset comprising EMNIST data and a custom collection of capital letters (A to Z). The primary objective is to accurately classify handwritten letters into their respective categories.

## File Structure:
 *sample_images:* This directory contains samples of handwritten letters utilized for both training and model evaluation.
*image_to_pixel.ipynb:* This Jupyter Notebook encompasses the preprocessing steps for handwritten data, converting it into a numerical format, and generating the capital_samples.csv file.
*tensorflow_model.ipynb:* Here resides the finalized CNN model, meticulously documented with comments for ease of understanding. It includes training and testing procedures.

## Training Approach:
1. *Dataset Integration:* Initially, the CNN model undergoes training on the widely recognized EMNIST dataset, incorporating a diverse range of handwritten letters and digits. This phase serves as a benchmark for evaluating model performance.
2. *Custom Dataset Expansion:* Subsequently, the model is further refined through training on a bespoke dataset, encompassing capital letters from A to Z. To bolster generalization, this dataset is enriched with augmented images.


## Future Enhancements:
*Word Dataset Integration:* To augment its capabilities, the model can be reinforced by training on longer word datasets. This enhancement aims to bolster its proficiency in recognizing complete words and phrases.

## Dependencies:
-Python 3.x
-TensorFlow/Keras
-NumPy
-Matplotlib
-scikit-learn

## Usage Instructions:
1. Clone the repository to your local machine.
2. Execute the image_to_pixel.ipynb notebook to preprocess handwritten data and generate the capital_samples.csv file.
3. Open and execute the tensorflow_model.ipynb notebook to commence training and evaluation of the CNN model.

## Contribution Guidelines:
Contributions to this project are encouraged! Whether it's refining the model's accuracy, incorporating new features, or enhancing documentation, your contributions are valuable. Please feel free to submit pull requests or open issues for any suggestions or encountered issues.

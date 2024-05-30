# Plant Disease Classification using TensorFlow and Keras

This project focuses on leveraging deep learning techniques with TensorFlow and Keras to classify plant diseases. By training a Convolutional Neural Network (CNN) on the PlantVillage dataset, we aim to accurately identify various diseases affecting plants based on leaf images.


## Dataset

The PlantVillage dataset is a comprehensive collection of plant leaf images, each associated with specific disease classes. It encompasses different plant species and a variety of diseases that commonly affect them.

- **Dataset Download**: You can obtain the PlantVillage dataset [here](https://www.kaggle.com/datasets/emmarex/plantdisease). Ensure that you organize the dataset as described in the project structure section.

## Project Structure

- **Dataset**: Place the PlantVillage dataset in a directory named "PlantVillage." Subdirectories within "PlantVillage" represent various disease classes.

- **train_test_split.py**: Use this script to split the dataset into training, validation, and testing sets. It employs shuffling and creates TensorFlow data pipelines for efficient model training.

- **train_model.py**: This script defines and trains the CNN model using TensorFlow and Keras. The model architecture comprises convolutional layers, max-pooling layers, and fully connected layers.

- **predict.py**: Utilize this script to make predictions using the trained model. It accepts an input image and provides the predicted disease class along with a confidence level.

- **evaluate_model.py**: This script evaluates the model's performance on a test dataset and displays the accuracy.

## Model Downloads

You can download the pre-trained models for plant disease classification from the following Google Drive link:

[Plant Disease Classification Models](https://drive.google.com/drive/folders/14k1bIcmd6kCqPx5QSMDHJ7QCh4b0s92p?usp=sharing)

Please note that these models are pre-trained on the PlantVillage dataset and can be used for making predictions on new plant disease images.


## Model Architecture

Our CNN model architecture incorporates the following layers:

- Input Layer: Resizes and normalizes input images.
- Data Augmentation: Randomly applies horizontal and vertical flips, as well as rotations, to augment the dataset.
- Convolutional Layers: Multiple convolutional layers with ReLU activation.
- Max-Pooling Layers: Max-pooling layers to reduce spatial dimensions.
- Fully Connected Layers: Dense layers with ReLU activation.
- Output Layer: A dense layer with softmax activation for multi-class classification.

## Training and Evaluation

The model is trained using the training dataset and validated using the validation dataset to prevent overfitting. After training, it undergoes evaluation on the test dataset to measure its accuracy and generalization performance.

## Usage

To successfully run this project:

1. Download the PlantVillage dataset and organize it as specified in the "Dataset" section.

2. Execute the `train_test_split.py` script to split the dataset into training, validation, and testing sets.

3. Run the `train_model.py` script to define and train the CNN model.

4. Use the `evaluate_model.py` script to assess the trained model's performance on the test dataset.

5. Employ the `predict.py` script to make predictions on new plant disease images.

## Results

The accuracy of the trained model on the test dataset is reported in the `evaluate_model.py` script. Additionally, the `predict.py` script allows you to visualize model predictions on test images.

## Conclusion

This mini-project showcases the power of deep learning in plant disease classification. By following the instructions provided in this README, you can develop and deploy a robust model for classifying plant images based on various disease categories.

Feel free to customize and expand upon this project to address your specific plant disease classification requirements. Happy coding!

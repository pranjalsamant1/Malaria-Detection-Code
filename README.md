# Malaria Cell Image Classification Using CNN

This project demonstrates the use of Convolutional Neural Networks (CNNs) for classifying malaria cell images as **Infected (Parasitized)** or **Uninfected**. It applies deep learning techniques for medical image analysis to automate the malaria diagnosis process.

---

## Key Features

- **Data Preprocessing**: Used `ImageDataGenerator` for data augmentation and splitting into training and validation datasets.
- **Model Architecture**: Designed a CNN model with multiple convolutional layers, max-pooling layers, and dropout layers to extract and classify features.
- **Training**: Compiled and trained the model using the Adam optimizer and binary crossentropy loss function, with early stopping to prevent overfitting.
- **Evaluation**: Visualized model accuracy and loss trends during training and validation phases.
- **Prediction**: Tested the trained model on unseen images to predict whether they are infected or uninfected.

---

## Steps Summary

1. **Data Loading and Visualization**:
   - Loaded malaria cell images and visualized sample images from both "Infected" and "Uninfected" categories.
   
2. **Data Augmentation**:
   - Applied rescaling and augmentation to improve generalization and split the dataset into training and validation sets.

3. **Model Building**:
   - Built a CNN model consisting of:
     - **Convolutional Layers** to extract image features.
     - **Max-Pooling Layers** to reduce spatial dimensions.
     - **Dropout Layers** to prevent overfitting.
     - **Dense Layers** for final binary classification.

4. **Training**:
   - Trained the model with 20 epochs using augmented training data, with early stopping based on validation loss.

5. **Evaluation and Prediction**:
   - Plotted learning curves for accuracy and loss.
   - Tested the model on a new image to predict the infection status.

---

## Tools and Skills

- **Python Libraries**:
  - TensorFlow/Keras for deep learning model design and training.
  - Matplotlib for visualizations.
  - OpenCV for image handling.

- **Skills**:
  - Building CNN architectures for image classification.
  - Data preprocessing and augmentation.
  - Evaluating model performance with training/validation metrics.
  - Deploying and testing models on unseen data.

---

## Conclusion

This project showcases how CNNs can be applied to medical image classification tasks. It highlights key skills in deep learning, such as model building, training, and evaluation. This approach could be further expanded for deployment in real-world malaria diagnosis systems.

Feel free to explore, fork, and contribute to this repository!

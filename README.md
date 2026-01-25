<h1>Plant Disease Classification Using ViT (Vision Transformer)</h1>

<p>This project aims to classify plant diseases from images using a pre-trained Vision Transformer (ViT) model. The dataset used consists of 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset has been divided into 80% for training and 20% for validation. The final model achieves an impressive accuracy of over 99.6%.</p>
<p><a href="https://colab.research.google.com/drive/1ssLcWrLvSAVq6YBFrcQLtez9fE80Tdxd?usp=sharing">Open plantDiseaseClassification.ipynb in Colab</a></p>

<h2>Dataset</h2>
<p>The dataset is available on <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset">Kaggle</a>. It is a recreation of the original dataset using offline augmentation. Additionally, a directory with 33 test images has been created for the prediction phase.</p>

<h2>Project Structure</h2>
<ul>
  <li><strong>Training and Validation:</strong> 80% of the dataset is used for training, while 20% is used for validation. The dataset is preprocessed using image augmentation techniques, including random resizing, horizontal and vertical flips, Gaussian blur, and normalization.</li>
  <li><strong>Model Architecture:</strong> We used the <code>google/vit-base-patch16-224-in21k</code> model, a pre-trained Vision Transformer model on the ImageNet-21k dataset. The model is fine-tuned for the classification task on the plant disease dataset.</li>
  <li><strong>Metrics:</strong> The model is evaluated based on accuracy, and training is performed for 20 epochs with a learning rate of 2e-5. Gradients are accumulated over 4 steps, and the best model is saved based on the accuracy metric.</li>
  <li><strong>Prediction:</strong> A small test set of 33 images is used for final predictions. The trained model is evaluated on this test set, and it predicts the correct class with over 99.6% accuracy.</li>
</ul>

<h2>Usage</h2>

<ol>
  <li>Download the dataset from <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset">Kaggle</a>:</li>
  <pre><code>!kaggle datasets download -d vipoooool/new-plant-diseases-dataset
!unzip new-plant-diseases-dataset.zip</code></pre>

  <li>Install the required dependencies:</li>
  <pre><code>!pip install evaluate accelerate datasets torch torchvision transformers</code></pre>

  <li>Run the Python script to train the model and evaluate its performance. The script performs the following key tasks:</li>
  <ul>
    <li>Loads and preprocesses the dataset using image augmentation techniques.</li>
    <li>Initializes the Vision Transformer (ViT) model with pre-trained weights from ImageNet-21k.</li>
    <li>Trains the model and evaluates its performance on the validation set at each epoch.</li>
    <li>Makes predictions on a test set and calculates the final accuracy.</li>
  </ul>
</ol>

<h2>Key Python Libraries</h2>
<ul>
  <li>Transformers (for using the Vision Transformer model)</li>
  <li>PyTorch (for model training)</li>
  <li>Evaluate (for loading metrics like accuracy)</li>
  <li>Datasets (for loading the dataset in an image folder format)</li>
</ul>

<h2>Results</h2>
<p>The model achieves an accuracy of over 99.6% on the test set of 33 images, indicating its strong capability in classifying plant diseases based on leaf images.</p>

<h2>Conclusion</h2>
<p>This project demonstrates how transfer learning with Vision Transformer models can be highly effective in solving image classification tasks.</p>

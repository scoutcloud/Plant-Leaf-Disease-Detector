\# Plant Leaf Disease Detector



A Computer Vision based project for detecting tomato leaf diseases using image processing and Machine Learning.



\## Project Overview



This project analyzes tomato leaf images and predicts whether the leaf belongs to one of three classes:



\- Healthy

\- Tomato Early Blight

\- Tomato Late Blight



The project uses Computer Vision techniques such as image preprocessing, leaf segmentation, disease-region detection, feature extraction, and Principal Component Analysis (PCA).



For disease classification, the project uses:



\- K-Nearest Neighbors (KNN)

\- Naive Bayes



The final model is selected based on validation performance.



\## Dataset



The project uses the PlantVillage dataset.



Only three tomato classes are used:



\- Tomato\_\_\_healthy

\- Tomato\_\_\_Early\_blight

\- Tomato\_\_\_Late\_blight



The dataset is divided into training, validation, and testing sets.



The dataset is not included in this repository because of its size.



\## Technologies Used



\- Python

\- OpenCV

\- NumPy

\- Matplotlib

\- Pandas

\- Scikit-learn

\- Pillow



\## Project Structure



plant-leaf-disease-detector/

│

├── main.py

├── predict.py

├── README.md

├── .gitignore

├── leaf.jpg

│

├── models/

│   ├── knn\_model.pkl

│   ├── naive\_bayes\_model.pkl

│   ├── pca.pkl

│   └── scaler.pkl

│

├── outputs/

│

└── src/

&#x20;   ├── dataset\_loader.py

&#x20;   ├── disease\_analysis.py

&#x20;   ├── evaluation\_visualization.py

&#x20;   ├── feature\_extraction.py

&#x20;   ├── knn\_classifier.py

&#x20;   ├── naive\_bayes\_classifier.py

&#x20;   ├── pca\_analysis.py

&#x20;   ├── segmentation.py

&#x20;   └── severity\_analysis.py



\## Installation



\### 1. Clone the repository



git clone https://github.com/scoutcloud/Plant-Leaf-Disease-Detector.git



cd Plant-Leaf-Disease-Detector



\### 2. Create the Conda environment



conda create -n plant\_cv python=3.11



conda activate plant\_cv



\### 3. Install the required libraries



pip install opencv-python numpy matplotlib pandas scikit-learn pillow



\## Running the Project



Place the dataset inside the project directory.



Then run:



python main.py



This loads the dataset, extracts features, applies PCA, trains KNN and Naive Bayes classifiers, and evaluates their performance.



\## Predicting a New Leaf



To predict the disease of a leaf image, run:



python predict.py leaf.jpg



The program performs:



1\. Leaf segmentation

2\. Disease region detection

3\. Affected-area estimation

4\. Severity estimation

5\. Feature extraction

6\. PCA transformation

7\. KNN classification



The prediction results and processed images are saved inside the outputs folder.



\## Results



The current experiment produced the following results:



| Model | Validation Accuracy | Test Accuracy |

|---|---:|---:|

| KNN | 88.78% | 89.07% |

| Naive Bayes | 78.78% | 79.35% |



KNN achieved better validation performance and was selected as the final prediction model.



PCA reduced the feature representation from 12 original features to 5 components.



\## Computer Vision Techniques



The project uses:



\- Image resizing

\- Grayscale conversion

\- Gaussian filtering

\- Histogram equalization

\- HSV-based leaf segmentation

\- Morphological operations

\- Disease-region detection

\- Feature extraction

\- PCA



\## Machine Learning



Two classifiers are implemented:



\### KNN



K-Nearest Neighbors is used to classify the leaf based on its extracted and PCA-transformed features.



\### Naive Bayes



Gaussian Naive Bayes is implemented as a second classifier for comparison.



\## Limitations



\- The current system supports three tomato leaf classes.

\- Disease-region detection uses color and intensity based image processing.

\- Affected-area and severity values are estimates.

\- Performance can vary depending on lighting, background, image quality, and leaf orientation.

\- The current system uses manually extracted features rather than deep learning features.



\## Future Improvements



\- Add more plant species and disease classes.

\- Improve disease-region segmentation.

\- Add more texture features.

\- Experiment with additional classifiers.

\- Use image augmentation.

\- Compare traditional Computer Vision with CNN-based methods.

\- Improve severity estimation using labelled disease-area data.



\## References



PlantVillage Dataset:



https://github.com/urayoanm/plantvillage-dataset



OpenCV:



https://opencv.org/



Scikit-learn:



https://scikit-learn.org/


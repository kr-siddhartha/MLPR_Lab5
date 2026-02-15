# Lab 5: Face Detection and K-Means Clustering Analysis
**Name:** Siddhartha Kumar  
**Student ID:** U20240019  
**Major:** CSAI
**Course:** Machine Learning and Pattern Recognition (MLPR)  
**Lab Session:** 5  


## Project Overview:
This project explores the intersection of **Computer Vision** and **Unsupervised Machine Learning**. The primary focus is to detect human faces from a group image and classify them into distinct clusters based on their skin tone and lighting features using the **K-Means Clustering** algorithm. The project demonstrates how raw image data can be processed into meaningful feature vectors (Hue and Saturation) to discover hidden patterns without labeled data. Finally, a "template" image (Dr. Shashi Tharoor) is introduced to test the model's ability to classify new, unseen data into the established clusters.


## Aim and Objectives:
The main aim of this laboratory experiment is to implement a pipeline for object detection and clustering.

**Specific Objectives:**
1.  **Face Detection:** Utilize OpenCV's pre-trained Haar Cascade classifiers to identify and extract faces from a high resolution group photograph.
2.  **Feature Engineering:** Convert images from BGR to HSV color space to extract robust features (Mean Hue and Mean Saturation) that are less sensitive to lighting variations than RGB.
3.  **Clustering:** Apply the K-Means algorithm to group the detected faces into $k=2$ distinct clusters.
4.  **Classification:** Predict the cluster membership of a new template image based on the learned centroids.
5.  **Visualization:** Create comprehensive scatter plots that map the faces in a 2D feature space.


## Methodology:

### 1. Data Acquisition & Preprocessing
The input data consists of a group photograph and a target template image. 
![Plaksha_Faculty](https://github.com/user-attachments/assets/3fd2a00f-a05c-4bbe-adaf-01febc3c07fb)  
![Dr_Shashi_Tharoor](https://github.com/user-attachments/assets/b1770b37-3c73-4e26-a99a-13314effefda)
* **Image Loading:** Images are loaded using `cv2.imread()`.
* **Color Conversion:** Since OpenCV loads images in BGR format by default, they are converted to RGB for correct visualization using `matplotlib`.

### 2. Face Detection (Haar Cascades)
To isolate the region of interest (faces), I utilized the `cv2.CascadeClassifier`.
* **Algorithm:** Haar Feature based Cascade Classifiers.
* **Process:** The classifier uses a sliding window approach to detect edge, line, and four rectangle features.

> **Note:** The detection logic relies on a pre-trained XML file (`haarcascade_frontalface_default.xml`) provided by the OpenCV library.

### 3. Feature Extraction (HSV Color Space)
Standard RGB values are highly correlated with light intensity. To perform accurate color-based clustering, the detected face regions were converted to the **HSV (Hue, Saturation, Value)** color space.
* **Hue (H):** Represents the dominant color family.
* **Saturation (S):** Represents the intensity or purity of the color.
* **Feature Vector:** For each face, we calculated the **Mean Hue** and **Mean Saturation** to represent that face as a single point $(x, y)$ in 2D space.

### 4. K-Means Clustering
The core machine learning task involved partitioning the $N$ faces into $k$ clusters.
* **Algorithm:** K-Means.
* **Clusters ($k$):** Set to 2.
* **Distance Metric:** Euclidean Distance.
* The algorithm iteratively minimizes the within-cluster sum of squares (WCSS) to find the optimal centroids for the two groups.

### 5. Template Classification & Visualization
To validate the model, a template image (Dr. Shashi Tharoor) was processed separately to test the clustering logic on unseen data.
* **Template Processing:** The template face was detected, cropped, and converted to HSV space to extract its specific Mean Hue and Saturation.
* **Prediction:** The extracted feature vector was passed to the trained K-Means model `kmeans.predict` to classify the template into Cluster 0 or Cluster 1.
* **Overlay Visualization:** The template image was then plotted onto the existing scatter plot with a distinct border and marker size to visually confirm its position relative to the calculated centroids.

## Key Findings & Visualizations

### Visual 1: Detected Faces
The Haar Cascade classifier successfully identified multiple faces within the group image. Bounding boxes were drawn to verify accuracy before feature extraction.

<img width="1280" height="854" alt="Total number of face detected are 30" src="https://github.com/user-attachments/assets/7232b098-e4fb-41ed-8703-1a5fcf840f75" />
*Figure 1: Result of Haar Cascade Face Detection on the faculty group photo.*

### Visual 2: K-Means Clustering Results
The scatter plot below visualizes the distribution of faces based on their Mean Hue (X-axis) and Mean Saturation (Y-axis).

<img width="1002" height="538" alt="Screenshot 2026-02-15 at 10 24 53 PM" src="https://github.com/user-attachments/assets/8124cabc-41a1-4789-adef-8a61662cd44c" />
*Figure 2: Scatter plot showing the clustering of faculty faces. The actual face images are plotted as markers to visualize the grouping.*

### Visual 3: Template Classification
* **Cluster 0 (Green Points):** Represents faces with specific lighting/skin-tone characteristics defined by Centroid 0.
* **Cluster 1 (Blue Points):** Represents the second group with distinct average Hue/Saturation values.
* **Centroids:** Marked with large 'X' markers, representing the mathematical center of each cluster.

<img width="1005" height="538" alt="Screenshot 2026-02-15 at 10 26 32 PM" src="https://github.com/user-attachments/assets/afb2183e-e302-42da-8fbc-d1daac32bba0" />
*Figure 3: Classification of the template image relative to the existing clusters.*

### Visual 4: Template Face Detection
The Haar Cascade classifier is applied to the raw input image of Dr. Shashi Tharoor to accurately detect and crop the face, ensuring that background colors do not interfere with the HSV feature calculation.

<img width="400" height="400" alt="Detected Face in Template_screenshot_15 02 2026" src="https://github.com/user-attachments/assets/f01da49e-b810-46c9-9b7f-9ea32ee5ff9d" />

*Figure 4: Successful detection of the face in the template image using Haar Cascades.*

### Visual 5: Template Integration into Feature Space
To visually verify the classification, the extracted template face is plotted onto the existing cluster graph alongside the faculty members. A distinct border is added to the template image marker to distinguish it from the training data, allowing us to observe exactly where its skin tone and lighting conditions place it within the existing groups.

<img width="1005" height="545" alt="image" src="https://github.com/user-attachments/assets/93354ae4-f8e4-4e7f-832b-aa31310c0ed1" />
*Figure 5: The template face plotted on the graph, showing its similarity to neighbors in the cluster.*

### Visual 6: Centroid Distance & Final Classification
This plot highlights the mathematical decision-making process. The violet marker represents the template's exact coordinate (Hue,Saturation). By calculating the Euclidean distance between this point and the two centroids (Red 'X' and Orange 'X'), the model assigns the class based on the shorter distance.

<img width="1005" height="545" alt="image" src="https://github.com/user-attachments/assets/f47574a2-6e9b-498e-a059-740835e7d5b8" />
*Figure 6: Final classification plot showing the template point (Violet) relative to the cluster centroids.*

## Conclusion
This lab successfully demonstrated the end-to-end pipeline of an image processing and machine learning project.
1.  **Robustness of HSV:** Using Hue and Saturation proved to be a reliable method for distinguishing between subtle variations in face images, more so than raw RGB pixel data.
2.  **Unsupervised Capabilities:** K-Means successfully found structure in unlabelled data, grouping faces based on objective mathematical proximity rather than semantic labels.
3.  **Scalability:** The pipeline allows for any new face (like the template image) to be instantly classified without retraining the entire model, simply by calculating its distance to the nearest centroid.

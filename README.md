This Intrusion Detection System (IDS) is designed to detect and prevent intrusions in Internet of Things (IoT) systems using a hybrid-unsupervised machine learning algorithm. The algorithm utilizes autoencoders and One-class Classifier to separate benign data from malicious data (specifically Mirai and gafgyt attacks), thereby preventing intrusions via data downloads. The model is trained with 27,000 inputs of benign data and managed to keep the error rate less than 2 percent after training before testing.

Features:
Hybrid-Unsupervised Machine Learning Algorithm
Utilizes Autoencoders and One-class Classifier
Detects and Prevents Intrusions in IoT Systems
Specifically Designed to Prevent Mirai and gafgyt Attacks
Low Error Rate (<2%) after Training

How it Works:
Data Collection: Collect data from IoT sensors and devices.
Preprocessing: Preprocess the data to prepare it for training.
Training: Train the hybrid-unsupervised machine learning algorithm using autoencoders and One-class Classifier. The model is trained with a dataset containing 27,000 inputs of benign data.
Testing: Test the trained model with incoming data streams to detect intrusions.
Prevention: Prevent intrusions by identifying and filtering out malicious data packets, specifically Mirai and gafgyt attacks.


Requirements
Python 3.x
TensorFlow
Scikit-learn
NumPy
Pandas

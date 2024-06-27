# Diabetes-Prediction-Using-Machine-Learning

## Introduction
This project aims to predict the onset of diabetes using various machine learning algorithms. By comparing the performance of different models, we identify the most effective method for this task.

## Dataset
The dataset contains medical details of patients, including features such as glucose level, blood pressure, insulin level, BMI, age, and more. The target variable indicates whether a patient has diabetes.

## Models Used
- Decision Tree
- Logistic Regression
- Support Vector Machine
- Naive Bayes
- Random Forest

## Prerequisites
Ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
<h2>Project Structure</h2>
<ul>
  <li><code>diabetes_prediction.ipynb</code>: Jupyter Notebook containing the entire project.</li>
  <li><code>diabetes.csv</code>: Dataset used for training and testing the models.</li>
  <li><code>requirements.txt</code>: List of required Python libraries.</li>
  <li><code>README.md</code>: Project documentation.</li>
</ul>

<h2>Getting Started</h2>
<p>Clone the repository:</p>
<pre><code>git clone https://github.com/Lara311/Diabetes-Prediction-Using-Machine-Learning.git</code></pre>

<p>Navigate to the project directory:</p>
<pre><code>cd Diabetes-Prediction-Using-Machine-Learning</code></pre>

<p>Install the required libraries:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<p>Open and run the Jupyter Notebook:</p>
<pre><code>jupyter notebook diabetes_prediction-using-machine-learning.ipynb</code></pre>

<h2>Running the Notebook</h2>
<p>Open and run the <code>diabetes_prediction-using-machine-learning.ipynb</code> notebook to:</p>
<ul>
  <li>Explore the dataset.</li>
  <li>Preprocess the data.</li>
  <li>Build multiple classification models.</li>
  <li>Evaluate the performance of each model.</li>
</ul>

<h2>Results</h2>
<h3>Decision Tree Classifier</h3>
<ul>
  <li>Accuracy: 74.68%</li>
  <li>Precision: 62.50%</li>
  <li>Recall: 72.73%</li>
  <li>F1 Score: 67.23%</li>
</ul>

<h3>Logistic Regression</h3>
<ul>
  <li>Accuracy: 75.32%</li>
  <li>Precision: 64.91%</li>
  <li>Recall: 67.27%</li>
  <li>F1 Score: 66.07%</li>
</ul>

<h3>Support Vector Machine</h3>
<ul>
  <li>Accuracy: 72.73%</li>
  <li>Precision: 63.27%</li>
  <li>Recall: 56.36%</li>
  <li>F1 Score: 59.62%</li>
</ul>

<h3>Naive Bayes</h3>
<ul>
  <li>Accuracy: 76.62%</li>
  <li>Precision: 66.10%</li>
  <li>Recall: 70.91%</li>
  <li>F1 Score: 68.42%</li>
</ul>

<h3>Random Forest</h3>
<ul>
  <li>Accuracy: 72.73%</li>
  <li>Precision: 61.82%</li>
  <li>Recall: 61.82%</li>
  <li>F1 Score: 61.82%</li>
</ul>

<h3>Best Performing Model</h3>
<p>The Naive Bayes model performed the best with an accuracy of 76.62% and an F1 score of 68.42%.</p>

<h2>Conclusion</h2>
<p>In this project, we explored and analyzed a diabetes dataset to predict the onset of diabetes using various machine learning algorithms. We found that the Naive Bayes classifier performed the best in terms of accuracy and F1 score.</p>

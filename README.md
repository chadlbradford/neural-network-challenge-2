
# 🧠 Neural Network Challenge 2 – Employee Attrition & Department Prediction

This project implements a **branched neural network** using TensorFlow/Keras to assist an HR team in two key predictive tasks:

1. **Attrition Prediction** – Estimate whether an employee is likely to leave the company.
2. **Department Recommendation** – Predict which department an employee is best suited for.

The model is trained on HR-related features and outputs predictions for both classification tasks using a dual-output architecture.

---

## 📁 Dataset

The dataset contains employee records, including demographic information, work history, satisfaction levels, and job details. It was loaded from:

**Source:**  
[Attrition Dataset (provided by course)](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv)

---

## ✅ Features Used

Ten features were selected for training the model:

- Age  
- DistanceFromHome  
- Education  
- JobSatisfaction  
- WorkLifeBalance  
- JobLevel  
- NumCompaniesWorked  
- TrainingTimesLastYear  
- YearsAtCompany  
- EnvironmentSatisfaction  

Target columns:
- `Attrition` (Yes/No, one-hot encoded)  
- `Department` (3 categories, one-hot encoded)

---

## ⚙️ Preprocessing

- Split into training and testing sets using `train_test_split`  
- Scaled features using `StandardScaler`  
- One-hot encoded `Attrition` and `Department` columns with `OneHotEncoder`  
- Verified that all features were numeric prior to scaling  

---

## 🧠 Model Architecture

The model was built using the **Functional API** in Keras:

- **Input Layer:** 10 input features  
- **Shared Layers:**  
  - Dense(64, relu)  
  - Dense(128, relu)  
- **Attrition Branch:**  
  - Dense(32, relu)  
  - Dense(2, softmax)  
- **Department Branch:**  
  - Dense(32, relu)  
  - Dense(3, softmax)  

The model was compiled with the `adam` optimizer and categorical crossentropy losses for both outputs. It was trained for 100 epochs with a validation split of 20%.

---

## 📊 Evaluation Results

After evaluation on the test set, the model achieved:

```
Attrition predictions accuracy: 0.7880434989929199  
Department predictions accuracy: 0.5
```

Note: The department data was class-imbalanced (e.g., "Human Resources" underrepresented), which may affect accuracy metrics. Future improvements may include resampling or weighted loss functions.

---

## 📌 Summary Insights

- **Accuracy** is a reasonable metric for department prediction if class balance is improved, but for attrition (a binary target with potential imbalance), precision/recall or F1-score might be more informative.
- Both output branches used `softmax` because the labels were one-hot encoded.
- Model improvements could include dropout regularization, class balancing, hyperparameter tuning, or using alternative architectures.

---

## 📂 Files in This Repository

- `attrition.ipynb`: Main Jupyter notebook containing all preprocessing, model training, and evaluation steps
- `README.md`: This file

---

## 📚 References

- Self-sourced code vetted with ChatGPT  
- TensorFlow/Keras and Scikit-learn documentation  
- Provided dataset from edX AI course

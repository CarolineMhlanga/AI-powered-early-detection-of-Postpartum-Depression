# AI-powered-early-detection-of-Postpartum-Depression
#This project demonstrates a machine learning pipeline to predict the risk of postpartum depression (PPD) using a combination of clinical metadata (e.g., age, BMI, EPDS scores) and sentiment analysis of clinical notes. The system leverages XGBoost, a powerful tree-based classifier, and includes model interpretability using SHAP.

Project Overview
Objectives:
Preprocess clinical data containing text and numerical features.

Use sentiment analysis to extract emotion-based features from clinical notes.

Train and evaluate an XGBoost classifier to predict postpartum depression (PPD).

Explain model decisions using SHAP values to understand feature importance.

Features Used
Feature	Description
age	Age of the postpartum patient
bmi	Body Mass Index
epds_score	Edinburgh Postnatal Depression Scale score
clinical_notes	Text-based notes from clinicians capturing emotional context
sentiment_score	Derived score from clinical_notes using VADER sentiment
ppd	Binary label: 1 = PPD present, 0 = no PPD

Dependencies
The code uses the following Python libraries:

bash
Copy
Edit
pandas, numpy, matplotlib, seaborn, shap, sklearn, xgboost, vaderSentiment
You can install missing dependencies using:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn shap scikit-learn xgboost vaderSentiment
Data Format
The dataset should be in CSV format and contain the following columns:

csv
Copy
Edit
age,bmi,epds_score,clinical_notes,ppd
28,24.3,13,"She feels sad and overwhelmed",1
32,26.1,5,"Patient is coping well",0
Note: Make sure to replace DATA_PATH = 'ppd_sample.csv' with the actual path to your dataset.

Pipeline Steps
1. Data Loading
python
Copy
Edit
df = pd.read_csv(DATA_PATH)
Loads the CSV into a DataFrame. Verifies essential columns and prints unique class distribution.

2. Sentiment Analysis of Clinical Notes
python
Copy
Edit
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['clinical_notes'].apply(lambda note: analyzer.polarity_scores(note)['compound'])
Uses VADER (a lexicon-based sentiment analyzer) to assign a compound score to each clinical note, capturing its emotional tone.

Score Type	Description
+1.0	Extremely positive sentiment
0.0	Neutral
-1.0	Extremely negative sentiment

This score is added as a numeric feature called sentiment_score.

3. Preprocessing
python
Copy
Edit
df = df.dropna(subset=['age', 'bmi', 'epds_score', 'sentiment_score', 'ppd'])
X = df[['age', 'bmi', 'epds_score', 'sentiment_score']]
y = df['ppd']
Drops rows with missing critical values.

Constructs:

X: feature matrix with 4 features

y: target variable (ppd)

4. Train/Test Split
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=2, random_state=42)
Performs a stratified split to ensure both classes are represented in train and test sets. The test_size=2 is designed to ensure there's a minimal yet valid test set in small datasets.

5. Model Training with XGBoost
python
Copy
Edit
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
Trains an XGBoost classifier, known for its speed and accuracy on tabular data, with log-loss as the evaluation metric.

6. Model Evaluation
python
Copy
Edit
print(classification_report(y_test, y_pred))
confusion_matrix(...)
Outputs:

Classification report (precision, recall, F1-score)

Confusion matrix with labels 'No PPD' and 'PPD', visualized with matplotlib.

7. Explainability with SHAP
python
Copy
Edit
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
SHAP (SHapley Additive exPlanations) is used to:

Understand how each feature contributes to predictions.

Generate a bar plot ranking features by importance.

Save the plot as "shap_feature_importance.png".

This ensures the model is not a black box and helps clinicians understand why certain patients were flagged as high risk.

Outputs
Confusion Matrix Plot: Shows correct and incorrect predictions.

SHAP Feature Importance Plot: Visualizes key features driving model decisions.

Console Output:

Unique class distribution

Classification metrics (accuracy, F1-score, etc.)

Example Use Case
A 28-year-old woman with:

BMI: 24.3

EPDS score: 13

Clinical note: “She feels sad and overwhelmed”

Would be scored via the sentiment model and used in prediction. A high EPDS + negative sentiment increases the risk score, and the model may flag the patient as PPD=1.

Final Notes:
The model is a proof of concept, not yet clinically validated.

Larger and more diverse datasets would improve generalization.

Future work may involve:

NLP with transformers for deeper note analysis

Time-series modeling for longitudinal tracking

Integration with electronic health records (EHRs)

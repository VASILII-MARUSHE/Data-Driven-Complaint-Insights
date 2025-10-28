# Data-Driven-Complaint-Insights
Classifying customer complaints using text clustering, PCA, and Random Forest. Achieved 0.85 AUC-ROC with SHAP insights.

## Project Overview
This project analyzes customer complaint data, a financial services company. The goal is to identify patterns in complaints, classify them into internal (credit-related) vs. market/customer reasons, and provide actionable insights to reduce complaints.

Key components:
- **Exploratory Data Analysis (EDA):** Investigated complaint distributions by team, date, hour, and weekday. Identified anomalies (e.g., spikes on specific dates) and temporal patterns (e.g., peaks at 08:00 and 13:00).
- **Text Preprocessing and Clustering:** Used embeddings (e.g., OpenAI or Sentence Transformers) and Agglomerative Clustering to group complaint reasons into two semantic clusters: "Customer/Market Reasons" (Cluster 0) and "Internal Credit Reasons" (Cluster 1).
- **Feature Engineering:** Calculated Mutual Information (MI) for feature selection, applied PCA to reduce multicollinearity (retaining 9 components), and interpreted components (e.g., PC1 as "total fuel cost").
- **Modeling:** Built a Random Forest classifier (tuned with GridSearchCV using F1-score) to predict complaint clusters. Achieved AUC-ROC of 0.85 on the test set.
- **Evaluation and Interpretation:** Used confusion matrix, SHAP values for feature importance, and visualizations to explain model decisions (e.g., impact of `dd_margin`, `dd_litres`, and hour of day).
- **Recommendations:** Suggestions like increasing staffing during peak hours, analyzing internal processes, and integrating ML for early warnings.


## Dataset
- The dataset includes complaint details (e.g., creation date, time, reasons, financial metrics like revenue and margins).
- Data is anonymized and not included in this repo due to sensitivity. Replace with your own or mock data for reproduction.

## Results
- **Model Performance:** AUC-ROC = 0.85 (test set). Confusion matrix shows good balance between true positives/negatives.
- **Key Insights:**
- Complaints peak at 08:00 (likely internal processes) and on business days.
- Features like `dd_margin` and `dd_litres` strongly influence internal complaints when near zero.
- SHAP analysis links PCA components back to original features for interpretability.
- **Business Impact:** Recommendations could reduce complaints by addressing operational bottlenecks.

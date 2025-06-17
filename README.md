# Telco Customer Churn Prediction
## 1 Introduction
### 1.1 Nature of the Business Problem 

In a competitive telecommunications industry, retaining existing customers is significantly more cost-effective than acquiring new ones (​Dahiya & Bhatia, 2015)​. High churn not only reduces revenue but also increases operational costs related to customer acquisition and onboarding. This report addresses the problem of customer churn by developing a predictive model to identify at-risk customers in advance. By leveraging available customer data, the goal is to provide Telco with actionable insights that support strategic decision-making for customer retention. 

### 1.2 Business Objectives and Goals 

A predictive model will be developed to identify customers at high risk of churning, enabling the company to: 

- Implement targeted and proactive retention strategies 

- Uncover key churn drivers 

- Reduce revenue loss 

- Improve customer experience and loyalty 

## 1.3 Available Data Resources 

The Telco Customer Churn dataset was selected from Kaggle, which contains 7,043 records and 21 features, focusing on customer attributes relevant to churn.

## 1.4 Limitations and Issues 

- The dataset covers only customers who left in the last month, limiting long-term churn analysis. 

- Customer location data is unavailable for geographic insights. 

- Churn reasons are not included, hindering cause-specific analysis. 

- Lacks service ratings, promotional offer history, and complaint records, which are key churn indicators. 

- Dataset has minor issues including blank strings in the ‘TotalCharges’ column (11 rows) and inconsistent category labels across several service-related columns. 

- The target variable (Churn) is imbalanced, which can affect model performance. 

## 2. Methodology 

### 2.1 Data Harvesting  

Dataset was sourced directly from Kaggle ​(Blastchar, 2018; Macko, 2019)​.  

### 2.2 Data Preprocessing 

- Dropped the ‘customerID’ column as it held no predictive value 

Replaced blank ‘TotalCharges’ entries (from new customers with tenure = 0) with 0.0 and converted the datatype to float. 

Standardized labels such as ‘No internet service’ and ‘No phone service’ to ‘No’. 

Categorical features were label-encoded for modeling. 

### 2.3 Exploratory Data Analysis (EDA) 

EDA was performed to understand data distributions, feature relationships, and churn drivers: 

A pie chart was plotted to visualize the class distribution of the target variable Churn. 

Numerical Feature Analysis 

Histograms were plotted for ‘Tenure’, ‘MonthlyCharges’, and ‘TotalCharges’ to inspect their distributions including mean and median lines for comparison. 

Boxplots were used to examine how numerical features vary by churn status, highlighting trends and outliers. 

A correlation heatmap was generated for numerical features to assess multicollinearity.  

Categorical Feature Analysis 
Stacked bar charts were used to visualize churn proportions across key categorical groups (e.g., demographics, services, billing methods), revealing churn drivers. 

### 2.4 Feature Selection 

A preliminary filter-based selection was conducted to reduce dimensionality and remove uninformative predictors prior to modeling. 

Chi-Squared test and ANOVA F-test ranked categorical and numerical features in relation to the target variable, Churn ​(Kuhn and Johnson, 2019)​.  

These tests ranked predictors by statistical relevance, enabling early exclusion of features weakly associated with churn. 

### 2.5 Train and Test Data Split 

Stratified 80:20 split preserved churn distribution across training and test sets.​  

### 2.6 Data Mining Technique Selection  

Table 1 shows the selection and justification of data mining techniques for churn prediction​.  

Technique  

Classification - Ideal for predicting categorical labels like churn. Provides probabilities and interpretable outputs.  

Structured data with predefined class labels. 

## 3. Implementation 

### 3.1 Model Development 

The following classification models were used ​(Han, Kamber and Pei, 2011)​​(Sarker, 2021)​:  

Decision Tree: Chosen for its simplicity, interpretability, and ability to handle both categorical and numerical features without requiring feature scaling. Its tree structure enables identification of feature importance. 

Random Forest: Selected for its robustness to outliers, feature collinearity, and class imbalance. As an ensemble of decision trees, it improves prediction accuracy and generalization while providing reliable feature rankings. 

XGBoost: Included for its superior performance in handling complex interactions, imbalanced classes, and noisy data through gradient boosting and regularization. It delivers high accuracy and ranks churn drivers effectively. 

Imbalance handling:  

class_weight='balanced' for Decision Tree and Random Forest ​(Pedregosa et al., 2011)​ 

scale_pos_weight for XGBoost ​(Chen and Guestrin, 2016)​ 

3.2 Evaluation Metrics  

Model performance was evaluated using ​(Han, Kamber and Pei, 2011)​: 

5-fold cross-validation (CV): Ensures robustness and generalizability of the model performance across different data splits. 

Accuracy: Measures the overall correctness of the model. 

Precision: Identifies how many predicted churns were actual churns. 

Recall: Reflects how many actual churns were correctly predicted. 

F1-Score: A harmonic mean of precision and recall, useful for imbalanced datasets. 

3.3 Hyperparameter Tuning 

To enhance model performance, ‘RandomizedSearchCV’ was used for hyperparameter tuning. This method efficiently samples hyperparameter combinations from predefined ranges to balance exploration and computational efficiency ​(Bergstra and Bengio, 2012)​. 

3.4 Final Model  

The final model was selected based on overall performance and practical applicability. 

It was trained on the full dataset using the best-tuned hyperparameters. 

Preprocessing steps (e.g., label encoding) applied during training were consistently applied to new input data to maintain format integrity. 

The trained model and encoders were saved using Python’s pickle module to enable reuse without retraining. 

The model’s ‘feature_importances_’ attribute was used to rank the most influential predictors and visualized using a bar plot for interpretability. 

New customer inputs were transformed using the saved encoders and fed into the trained model. The output included both a binary churn prediction and a probability score. 

A sample prediction output is provided in Appendix B to illustrate the model’s deployment. 

## 4 Results
Repository Contents

- `Customer_Churn_Prediction.ipynb`: End-to-end notebook including:
  - Data preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature selection (Chi-Square, ANOVA F-test)
  - Model training (Decision Tree, Random Forest, XGBoost)
  - Cross-validation and evaluation
  - Hyperparameter tuning
  - Business insights

---

5.  Conclusions 

5.1 Summary of Key Outcomes 

EDA revealed class imbalance of Target (Fig1), multicollinearity (Fig3), and outliers (Fig4), making linear models less suitable. Tree-based models (Decision Tree, Random Forest, XGBoost) were selected to address these challenges (Detailed in 3.1). A stratified 80:20 split was then applied to preserve churn distribution across training and test sets. Model evaluation emphasized precision, recall, and F1-score. 

EDA also revealed strong churn associations: 

Numerical: Tenure, MonthlyCharges, TotalCharges (Fig9) 

Categorical: Contract, InternetService, PaymentMethod (Fig10) 

Weak predictors such as Gender and PhoneService were removed before modeling. 

 

Random Forest delivered the highest accuracy and precision, supporting cost-effective retention strategies. XGBoost yielded highest recall and F1-score, excelling at identifying more churners (Table 02). 

Hyperparameter tuning improved recall and F1-score for both models (Table 03). 

Random Forest identified MonthlyCharges, TotalCharges, Tenure, and Contract as key churn drivers. XGBoost ranked Contract highest with InternetService as the second best following at a much lower impact (Fig11). 

5.2 Business Application of Final Model 

The chosen classification approach effectively predicts binary churn outcomes, providing clear probabilities for strategic decision-making. 

The two selected models, Random Forest and XGBoost, efficiently handled categorical and numerical features despite feature correlation and class imbalance. This makes them well suited for real-world customer data. 

Targeted retention: Random Forest’s high precision helps reduce unnecessary outreach by focusing on customers most likely to churn, maximizing retention return on investment. 

Proactive outreach: XGBoost’s higher recall makes it suitable when the priority is to retain as many potential churners as possible, even false positives. 

Both models also reveal key churn drivers, like contract type, internet service, and charges. Actionable recommendations based on these are outlined in Appendix C. 

To operationalize the model, it can be integrated into the company’s Customer Relationship Management system to flag high-risk customers in real time ​(Narapureddy, 2024)​. These risk scores can trigger targeted retention actions like personalized discounts, loyalty rewards, or proactive outreach. 

By aligning predictions with targeted actions, the company can retain more customers, reduce acquisition costs, and foster long-term loyalty, directly supporting the project’s initial business goals. 

Actionable Business Insights

| Churn Driver       | Insight | Recommended Business Action |
|--------------------|---------|------------------------------|
| **Internet Service** | Fiber optic users churn more than DSL or ‘No Internet’. | Investigate service quality. Improve speed and reliability. Set clearer expectations. |
| **Premium Services** | Customers without add-ons like ‘OnlineSecurity’ churn more. | Promote premium services via campaigns or bundled deals. |
| **Contract Type**    | Month-to-month users churn more than long-term contracts. | Incentivize switch to longer-term plans with discounts or loyalty points. |
| **Billing Method**   | Paperless billing users show higher churn. | Improve email notifications and digital bill visibility. |
| **Payment Method**   | Highest churn seen with electronic check users. | Promote auto-pay or app-based payment options. |
| **Tenure & Charges** | Early-life customers with high monthly charges are more likely to churn. | Offer welcome deals and proactive support in the first few months. |

---
## How to Run

Clone this repository:
```bash
[git clone https://github.com/erandime/Customer-Churn-Prediction.git]

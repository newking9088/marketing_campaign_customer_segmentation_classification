# marketing_campaign_customer_segmentation_classification
This repository contains a machine learning project to optimize marketing strategies for bank term deposits. Using models like Logistic Regression, Random Forest, SVM, KNN, and XGBoost, it classifies customers based on their likelihood of subscribing. 

# Enhance Marketing Campaign: Optimizing Term Deposit Subscriptions with 2-Fold Strategies of Segmentation and Classification

## Overview

A term deposit or certificate of deposit (CD) is a type of savings account offered by financial institutions. It features a lock-up period ranging from months to several years and provides a better interest rate than traditional bank savings accounts. Despite the fact that Certificates of Deposit (CDs) are insured up to $250,000 USD when deposited in Federal Deposit Insurance Corporation (FDIC) insured banks, and short-term CDs offered interest rates exceeding 6% in 2023, the prevalence of CD accounts remained comparatively low in contrast to savings accounts. According to a 2023 Forbes Advisor survey, only 3% of people have never opened a savings account, whereas 41% have never opened a CD.

To craft a marketing strategy that taps into the huge potential market for Certificates of Deposit (CDs), we studied Portuguese bank data collected from 2008 to 2012. Our goal was to reduce marketing campaign costs and optimize revenue by providing the best return products for customers. Instead of contacting all customers or doing so randomly, we aimed to identify potential customers who are more likely to subscribe to a CD. This approach ensures we offer the best value to customers while also increasing our revenue.

We used customer segmentation and binary classification techniques, employing models such as Logistic Regression, Random Forest Classifier, K-Neighbors Classifier, XGBoost Classifier, Voting Classifier, and Neural Networks to predict whether a customer will subscribe to a CD.

The hyper-tuned XGBoost model performed the best, achieving a recall of 77%, meaning that out of all the actual subscribers, our model correctly identified 77% of them. It also achieved an F1 score of 61.2%, indicating a good balance between precision and recall, especially given the highly imbalanced nature of our data. This performance was achieved with an optimized probability threshold of 0.29.

In this project, we will lay out recommendations and strategies to boost marketing campaigns based on these data-driven results.

## Background

Term deposits, also known as Certificates of Deposit (CDs), can be an excellent alternative for secure investments, offering higher interest rates compared to traditional savings accounts. The trade-off for the higher interest rate and no maintenance fee is that CDs charge a penalty for early withdrawal if you withdraw money before the lock-up period ends (maturity date). In contrast, traditional savings accounts typically allow up to six convenient monthly withdrawals before charging a penalty, though this may vary by bank. Short-term CDs (less than a year) generally provide lower interest rates than long-term CDs (more than a year). Customers typically choose between short-term and long-term CDs based on their financial needs.

In 2023, CDs yielded returns of over 6%. While the S&P500 Annual Return is over 7%, it comes with market volatility and the risk of capital loss. In contrast, CDs provide a secure return, especially appealing when federal interest rates are high. The average year-to-year inflation is over 2%. This can be a good product for customers who would want returns above the inflation rate and yield some returns securely.

## Motivation

According to a 2023 Forbes Advisor survey, only 3% of people have never opened a savings account compared to 41% who have never opened a CD. However, the survey also shows that 41% of people opening high-yield savings accounts did so to take advantage of recent interest rate hikes, compared to only 31% of those who opened CDs. This indicates that despite CDs offering fixed higher interest rates compared to the fluctuating lower rates of savings accounts, people are more likely to open savings or high-yield savings accounts than CDs.

Understanding why people are more inclined to open savings accounts rather than Certificates of Deposit (CDs) is crucial for developing effective marketing strategies. This raises important questions: Is the lack of interest in CDs due to a lack of product knowledge, ineffective marketing, or simply a preference for more flexible savings options? To address these questions, we should explore our data to identify the characteristics of potential CD customers and understand the barriers preventing others from considering CDs.

For instance, educational marketing campaigns could target customer groups unfamiliar with CDs, such as younger customers or those primarily using digital banking services, by using channels like social media, online ads, and email newsletters to highlight the security and higher interest rates of CDs.

Additionally, product subscription campaigns should focus on risk-averse investors, retirees, and conservative savers, emphasizing the stability and guaranteed returns of CDs, especially in a high-interest-rate environment. Personalized marketing through direct mail, in-person demos, and financial advisor recommendations can effectively reach these groups. Conversely, identifying customer groups who prefer high-risk, high-reward investments, such as active stock market traders, can help avoid targeting them with CD promotions, instead offering them more suitable products like investment opportunities.

## Methodologies

We collected data from the UCI Machine Learning Repository and performed several preprocessing steps, including handling missing values, encoding categorical variables, normalizing numerical features, model selection, cross-validation, and hyper-parameter tuning.

### Handling Missing Data

Missing values in categorical variables were replaced with ‘unknown’. The dataset is imbalanced, with 11.3% of customers subscribing to CDs and 88.7% not subscribing. The dataset contains 41,188 rows and 21 columns:

- **age:** The age of the customer.
- **job:** The type of job the customer has.
- **marital:** The marital status of the customer.
- **education:** The level of education of the customer.
- **default:** Indicates if the customer has credit in default.
- **housing:** Indicates if the customer has a housing loan.
- **loan:** Indicates if the customer has a personal loan.
- **contact:** The type of communication contact used.
- **month:** The last contact month of the year.
- **day_of_week:** The last contact day of the week.
- **duration:** The duration of the last contact in seconds.
- **campaign:** The number of contacts performed during this campaign and for this client.
- **pdays:** The number of days that passed by after the client was last contacted from a previous campaign.
- **previous:** The number of contacts performed before this campaign and for this client.
- **poutcome:** The outcome of the previous marketing campaign.
- **emp.var.rate:** Employment variation rate - quarterly indicator.
- **cons.price.idx:** Consumer price index - monthly indicator.
- **cons.conf.idx:** Consumer confidence index - monthly indicator.
- **euribor3m:** Euribor 3 month rate - daily indicator.
- **nr.employed:** Number of employees - quarterly indicator.
- **y:** Indicates whether a customer subscribed to CDs.

### Customer Segmentation

We divided the features into two segments:

**Demographic Segmentation:** ‘age’, ‘job’, ‘marital’, ‘education’

The demographic analysis shows that customers aged 30-70 are the most likely to subscribe to CDs. It indicates that retirees, as well as individuals working in administrative and technician roles, are the most likely to subscribe to CDs. In contrast, blue-collar workers are less likely to subscribe. Additionally, single individuals and those with university or professional degrees show a higher likelihood of subscribing to CDs.

**Behavioral Segmentation:** ‘default’, ‘housing’, ‘loan’, ‘poutcome’, ‘campaign’, ‘pdays’, ‘previous’

Customers with no default history are the most likely to subscribe to CDs. Housing loans have no significant effect on the likelihood of subscribing. People who have subscribed before have over a 65% chance of subscribing again. Additionally, customers who were contacted recently have a higher likelihood of subscribing to CDs. The likelihood of subscribing to CDs decreases exponentially with the number of calls made. Most customers who are likely to subscribe do so within the first few contacts. After 15 contacts, the likelihood of subscription almost drops to zero.

### Categorical Feature Transformation

Categorical variables with rare subcategories were adjusted:
- **‘housing’, ‘default’, ‘loan’, ‘marital’:** Subcategories with low frequency were merged with the dominant category.
- **‘job’, ‘education’:** Subcategories with less than 5% frequency were combined into a new category named ‘Others’. For predictive modeling, the categories ‘illiterate’ and ‘unknown’ will be merged into a new category labeled ‘Others’.

We assessed the association of categorical features with the target variable and among themselves using Cramer’s V score and the chi-square test. Due to the data being collected over only 10 months, with Q1 and Q4 having lower proportions, we grouped the data into Q2, Q3, and ‘Others’. The ‘day_of_week’ feature, which showed a weak association with the target, was dropped to avoid sparsity in the model.

### Encoding Categorical Features

Categorical features were one-hot encoded, and one random subcategory was dropped using scikit-learn’s `OneHotEncoder`.

### Handling Imbalanced Data

To address class imbalance, we employed several techniques:

- **Oversampling the Minority Class:** We duplicated the minority class observations in the training dataset to balance it with the majority class.
- **Class Weight Adjustment:** We assigned higher weights to the minority class during model training.
- **Threshold Tuning:** The probability threshold for determining crisp labels was fine-tuned, rather than using the default threshold of 0.5.

For models such as Logistic Regression, Random Forest, SVM, KNN, and XGBoost, we used `class_weight='balanced'` provided by the scikit-learn library. For Neural Networks, we used two approaches: optimizing class weights using GridSearch and upsampling the minority class.

### Model Selection, Evaluation, and Hyperparameter Tuning

We experimented with the following machine learning models:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **XGBoost Classifier**
- **Voting Classifier**
- **Neural Network**

#### Evaluation Metrics

The models were evaluated using a various set of metrics, including accuracy, precision, recall, F1 score, and ROC AUC. Since our goal was to accurately predict whether a customer would subscribe to CDs, we focused on optimizing recall while maintaining the highest possible F1 score.

#### Hyperparameter Tuning

We performed hyperparameter tuning using Grid Search with Cross-Validation to identify the optimal parameters for the best model, selected based on its ability to achieve optimal recall and F1 score, and generalize well to unseen data.

## Discussion

### Model Performance

We trained several models and evaluated them using the following metrics:

<table>
<caption style="caption-side: top; text-align: center;">Table 1: Evaluation metrics of select classifier models.</caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Train Time (sec)</th>
      <th>Train Score</th>
      <th>Test Score</th>
      <th>Train Precision</th>
      <th>Test Precision</th>
      <th>Train Recall</th>
      <th>Test Recall</th>
      <th>Test F1 Score</th>
      <th>Test ROC AUC Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic</td>
      <td>0.1610</td>
      <td>0.8381</td>
      <td>0.8371</td>
      <td>0.3934</td>
      <td>0.3914</td>
      <td>0.8051</td>
      <td>0.8040</td>
      <td>0.5265</td>
      <td>0.9017</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>0.9669</td>
      <td>0.8505</td>
      <td>0.8488</td>
      <td>0.4207</td>
      <td>0.4160</td>
      <td>0.8681</td>
      <td>0.8478</td>
      <td>0.5582</td>
      <td>0.9238</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>125.6661</td>
      <td>0.8453</td>
      <td>0.8361</td>
      <td>0.4149</td>
      <td>0.3957</td>
      <td>0.9099</td>
      <td>0.8628</td>
      <td>0.5426</td>
      <td>0.9144</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.0238</td>
      <td>0.9249</td>
      <td>0.8970</td>
      <td>0.7512</td>
      <td>0.5704</td>
      <td>0.4992</td>
      <td>0.3468</td>
      <td>0.4314</td>
      <td>0.8363</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>0.3031</td>
      <td>0.9539</td>
      <td>0.9103</td>
      <td>0.8501</td>
      <td>0.6236</td>
      <td>0.7170</td>
      <td>0.5140</td>
      <td>0.5636</td>
      <td>0.9400</td>
    </tr>
    <tr>
      <td>Hypertuned XGBoost</td>
      <td>0.4043</td>
      <td>0.9219</td>
      <td>0.9109</td>
      <td>0.6956</td>
      <td>0.6333</td>
      <td>0.5455</td>
      <td>0.4964</td>
      <td>0.5565</td>
      <td>0.9447</td>
    </tr>
    <tr style="background-color: #d0e7f9; font-weight: bold;">
      <td>Threshold-Tuned XGBoost*</td>
      <td>0.4043</td>
      <td>0.9219</td>
      <td>0.9109</td>
      <td>0.5710</td>
      <td>0.5538</td>
      <td>0.7883</td>
      <td>0.7721</td>
      <td>0.6623</td>
      <td>0.9446</td>
    </tr>
    <tr>
      <td>Hypertuned Voting Classifier</td>
      <td>2.3032</td>
      <td>0.9296</td>
      <td>0.9001</td>
      <td>0.6487</td>
      <td>0.5467</td>
      <td>0.8181</td>
      <td>0.7082</td>
      <td>0.6169</td>
      <td>0.9354</td>
    </tr>
    <tr>
      <td>Hypertuned Neural Network</td>
      <td>76.1232</td>
      <td>0.8157</td>
      <td>0.7952</td>
      <td>0.6123</td>
      <td>0.3412</td>
      <td>0.9045</td>
      <td>0.8793</td>
      <td>0.7303</td>
      <td>0.4917</td>
    </tr>
  </tbody>
</table>

Figure 5 below illustrates the feature importance based on odds ratios for logistic regression.

<figure id="attachment_99168" style="width: 768px; text-align: center;">
  <a href="https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2024/09/nawaraj-paudel-phd/feature-importance-138960-PqYPspiJ.png">
    <img src="https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2024/09/nawaraj-paudel-phd/feature-importance-138960-PqYPspiJ-768x521.png" alt="Feature importance" width="768" height="521" class="size-medium_large wp-image-99168" />
  </a>
  <figcaption>Figure 5: Feature importance using logistic regression.</figcaption>
</figure>

Figure 5 shows that call duration emerges as the most significant feature, followed by the consumer confidence index, single (not married) status, retirees, and the success of previous outcomes. The least significant feature is the method of contact. These feature importance align well with expectations, highlighting the key factors influencing the model’s predictions. The features ‘poutcome_nonexistent’, ‘previous’, ‘loan_yes’, and ‘housing_yes’ were not statistically significant at the 5% level and showed a very weak association with the target variable. Therefore, these features were dropped from the model.

Our goal was to predict whether a customer will subscribe to CDs, focusing on optimizing recall while maintaining a balance with precision to achieve a higher F1 score. Among the models we trained, XGBoost stands out by providing a comparable F1 score to other models but with a better balance between precision and recall, and it generalizes well to unseen data. Therefore, we will proceed with hyperparameter tuning for the XGBoost model. Although we also hyper-tuned voting classifiers and neural networks, they underperformed compared to the XGBoost model.

The XGBoost model was fine-tuned using `GridSearchCV`, resulting in optimal parameters: {‘colsample_bytree’: 1.0, ‘gamma’: 0.5, ‘max_depth’: 4, ‘min_child_weight’: 5, ‘subsample’: 1.0}, achieving a ROC-AUC score of 0.943. Further optimization of the probability threshold significantly improved the model’s performance on unseen data, boosting recall to 77.21%, precision to 55.4%, and the F1 score to 0.6623. Given the recall of 77.21%, the model correctly identifies 77.21% of actual subscribers. The remaining 22.79% of actual subscribers are missed. Precision of 55.4% indicates that out of all customers predicted as subscribers, 55.4% are correct, and the rest are false positives. Figure 6 presents the confusion matrix for the unseen data.

<figure id="attachment_99177" style="width: 746px; text-align: center;">
  <a href="https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2024/09/nawaraj-paudel-phd/cm-814768-BDnYMbg8.png">
    <img src="https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2024/09/nawaraj-paudel-phd/cm-814768-BDnYMbg8.png" alt="Confusion Matrix" width="746" height="601" class="size-full wp-image-99177" />
  </a>
  <figcaption>Figure 6: Confusion Matrix for Hypertuned and Threshold-Optimized XGBoost Model.</figcaption>
</figure>

### Impact of Data Balancing Techniques

Given that only 11.3% of customers subscribed to CDs, our data is heavily imbalanced, leading to a bias towards the majority class. To address this, we trained our models using balanced class weights, heavily penalizing errors in the minority class. For the Neural Network, class weights were optimized through Grid Search, and we also employed upsampling of the minority class. While the Neural Network performed well on the training set, it significantly underperformed on the test set compared to the XGBoost model. The performance of the XGBoost model was further improved through probability threshold optimization. The optimized probability threshold was 0.29.

### Challenges and Limitations

Despite the improvements, some challenges remain, such as the potential for overfitting and the need for continuous model updates to adapt to changing customer behavior. To address these challenges, we can continuously update the model with new data to capture evolving customer behavior and incorporate additional relevant features like bank balance, savings account status, and other financial behaviors. Including factors such as CD yield percentage and savings yield percentage can also enhance the model’s accuracy and relevance. This dataset includes only the month and day of the week, but not the year. For temporal data, it is recommended to split the training, validation, and test sets based on time to better mimic real-life scenarios and predict future outcomes. The earliest temporal data should be used for training and validation, while the later data should be reserved for testing. This approach ensures that the model is evaluated on its ability to generalize to future data.

## Recommendations

Different marketing strategies should be developed for customers with varying likelihoods of subscribing to CDs. We have segmented customers into three categories based on their probability of subscription and divided them into three tiers:

### Tier 1: High Probability (0.8 - 1.0) Group

These customers have a high likelihood of subscribing to CDs. We recommend the following strategy for Tier 1:

- **Personalized Offers:** Use the customer’s age, job, and education level to tailor offers. For example, offer higher interest rates or exclusive benefits to retirees or professionals.
- **Direct Communication:** Utilize the preferred contact method (e.g., telephone) to reach out directly with personalized messages.
- **Exclusive Deals:** Highlight the benefits of subscribing now, such as limited-time offers or bonuses for early subscribers.
- **Financial Advisory:** Offer personalized financial advice sessions to help them understand the benefits of CDs and how it fits into their financial plans.

### Tier 2: Medium Probability (0.5 - 0.8) Group

These customers have a moderate likelihood of subscribing to CDs. We recommend the following strategy for Tier 2:

- **Targeted Marketing Campaigns:** Use digital marketing and mobile banking apps to send targeted ads and notifications about CD benefits.
- **Educational Content:** Provide educational videos and articles on the banking homepage about the advantages of CDs and how they compare to other savings options.
- **Incentives:** Offer incentives such as small bonuses or interest rate boosts for subscribing within a certain period.
- **Quarterly Promotions:** Leverage the quarter information to run seasonal promotions that align with their financial planning cycles.

## Tier 3: Low Probability (0.3 to 0.5) Group

These customers have a low likelihood of subscribing to CDs. We recommend the following strategy for Tier 3:

- **Awareness Campaigns:** Use broad marketing strategies to raise awareness about CDs, focusing on their safety and reliability as an investment.
- **General Promotions:** Offer general promotions that appeal to a wider audience, such as introductory rates or flexible terms.
- **Cross-Selling:** Promote other banking products that might be of interest, such as savings accounts or loans, and subtly introduce CDs as a complementary product.
- **Customer Engagement:** Engage with these customers through surveys or feedback forms to understand their financial needs and tailor future offers accordingly.

## Practical Applications

Banks can use these predictive models to:

- **Targeted Marketing:** Develop personalized marketing campaigns to attract potential customers.
- **Resource Allocation:** Optimize resource allocation by focusing efforts on high-probability customers.

## Future Work

For future work, we could enhance our model by integrating additional data sources, such as customer transaction history, investment activities, and savings account details, to improve prediction accuracy. Additionally, incorporating temporal data more effectively by including the year and splitting the dataset based on time can help mimic real-life scenarios and predict future outcomes more accurately. Continuously updating the model with new data and incorporating relevant features like bank balance, savings account status, and financial behaviors will also be crucial in adapting to changing customer behavior and maintaining model performance.

Moreover, reaching out to customers through digital marketing, mobile banking apps, educational videos on the banking homepage, and personal financial advisory services can further enhance engagement and subscription rates. These strategies will not only help in identifying potential subscribers but also in providing them with valuable information and personalized financial advice, thereby increasing the likelihood of CD subscriptions.





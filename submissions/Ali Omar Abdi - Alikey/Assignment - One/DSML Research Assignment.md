# Introduction to Data Science and Machine Learning
> Research Assignment - DSML - 2026 


---

**Course:** Introduction to Data Science and Machine Learning  
**Due:** Monday, June 15, 2026 at 12:00 PM (Africa/Mogadishu / EAT)  
**Sources:** 16 references — books, scholarly articles, and peer-reviewed papers

**Author:** Ali Omar Abdi - Alikey


---

## Table of Contents

1. [Q1 — Data Science and Machine Learning: Definitions and Relationship](#q1----data-science-and-machine-learning-definitions-and-relationship)
2. [Q2 — The Data Science Lifecycle](#q2----the-data-science-lifecycle)
3. [Q3 — Supervised vs. Unsupervised Learning](#q3----supervised-vs-unsupervised-learning)
4. [Q4 — Overfitting: Causes and Prevention](#q4----overfitting-causes-and-prevention)
5. [Q5 — Training and Test Data Split](#q5----training-and-test-data-split)
6. [Q6 — Case Study: ML in Healthcare](#q6----case-study-ml-in-healthcare)
7. [References](#references)

---

## Q1 — Data Science and Machine Learning: Definitions and Relationship

### What Is Data Science?

Data Science is a multidisciplinary field that applies scientific methods, statistics, and software tools to extract useful knowledge from structured and unstructured data. It draws on computer science, mathematics, and domain knowledge to turn raw data into information that organizations can act on.

IBM (2025) describes data science as the field that "brings structure to big data" — covering the full pipeline from collection and cleaning through analysis and communication of findings. It is not defined by any single method; it is the overarching discipline that manages the complete data processing process.

### What Is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence where computer systems learn from data instead of following rules written explicitly by a programmer. As Pangeanic (2023) describes it, ML uses algorithms that "extract data, learn from these data, and generate forecasts about future trends."

The programmer does not write the rules. The algorithm works them out by processing examples.

### The Relationship Between Them

Data Science and Machine Learning are related but not the same thing. ML is one technique used within the broader work of Data Science. Sartorius (2020) explains that "Data science is a field that incorporates some areas of AI, machine learning and deep learning, while having a specific focus of gaining insight from data."

- **Data Science** is the end-to-end process: defining the question, gathering and cleaning data, exploring patterns, building models where needed, and presenting findings.
- **Machine Learning** is the modeling step used when the goal requires predicting outcomes or automating decisions.

Not every data science project uses ML. A data scientist may spend most of their time collecting, cleaning, and visualizing data with no model built at all. ML becomes relevant when the task involves learning patterns from historical data and applying them to new, unseen cases.

| Dimension | Data Science | Machine Learning |
|---|---|---|
| Scope | Broad, end-to-end process | Focused on learning algorithms |
| Goal | Extract insight and inform decisions | Build models that predict or automate |
| Techniques | Statistics, visualization, ML, data engineering | Supervised, unsupervised, reinforcement learning |
| Output | Reports, dashboards, predictions | Trained models, predictions |

### Real-Life Example: Credit Card Fraud Detection

Credit card fraud detection is one of the clearest examples of Data Science and Machine Learning working as a single system, each doing what the other cannot.

The **Data Science side** handles everything before the model exists. A bank collects years of transaction records — hundreds of millions of rows covering purchase amounts, merchant categories, locations, timestamps, and card-present vs. card-absent flags. That raw data is messy: duplicate records, missing merchant codes, transactions logged in different currencies and time zones. Data scientists clean it, standardize it, and engineer features that carry predictive signal. A raw timestamp, for instance, becomes "hour of day," "day of week," and "time since last transaction" — three separate features that a model can actually learn from. They also explore the data: what does a normal spending pattern look like? How do fraud cases differ in amount, frequency, and geography? These questions are answered before any model is built.

The **Machine Learning side** takes over once the data is ready. A supervised classification model — commonly a Random Forest or a Gradient Boosting model — is trained on historical transactions already labeled as fraud or legitimate. It learns which combinations of features (a large foreign transaction two minutes after a domestic one, for example) are statistically associated with fraud. As Syracuse University's iSchool (2025) notes, "machine learning enhances the actual process of fraud detection by recognizing subtle patterns that might indicate suspicious behavior" — patterns too complex and too numerous for a human analyst to monitor manually.

After deployment, the ML model scores every incoming transaction in milliseconds. The bank's data scientists continue monitoring it: tracking false positive rates, retraining the model as fraud tactics evolve, and adjusting decision thresholds based on the tradeoff between blocking legitimate customers and catching actual fraud.

Neither field works without the other here. Without the Data Science pipeline, there is no clean, structured dataset for the model to learn from. Without Machine Learning, there is no way to scale detection to millions of daily transactions. The two are not the same discipline — but in practice, they depend on each other at every stage.

| Stage | Who Does It | What Happens |
|---|---|---|
| Data collection | Data Science | Transaction records gathered from banking systems |
| Cleaning & engineering | Data Science | Missing values fixed, features like "time since last transaction" created |
| Exploration | Data Science | Normal vs. fraud spending patterns analyzed |
| Model training | Machine Learning | Random Forest trained on labeled fraud/non-fraud examples |
| Deployment | Both | Model integrated into transaction processing pipeline |
| Monitoring & retraining | Both | Performance tracked, model updated as fraud tactics change |

---

## Q2 — The Data Science Lifecycle

The Data Science lifecycle is the series of stages a data science project goes through, from the initial problem definition to deployment and ongoing monitoring. The most widely used framework is **CRISP-DM** (Cross-Industry Standard Process for Data Mining), developed in the late 1990s by a consortium of companies including Daimler-Benz and NCR, in consultation with over 200 practitioners. Excelsior University describes it as "the systematic framework for conducting successful data science projects," keeping technical work tied to real business needs at every stage.

CRISP-DM has six phases. They are not strictly sequential — projects often revisit earlier stages as new information emerges.

```
Business Understanding --> Data Understanding --> Data Preparation
        |                                                |
    Deployment <-- Evaluation <-- Modeling (ML lives here)
```

### Phase 1: Business Understanding

The project starts by defining the problem clearly. What does the organization need to learn? What decision will the results feed into? Studocu's CRISP-DM overview identifies vague business requirements as one of the most common reasons data science projects fail. A well-defined question is not optional — it determines everything that follows.

### Phase 2: Data Understanding

The team identifies what data is available and whether it is relevant to the problem. This means exploring datasets, checking data quality, and judging whether what exists is sufficient to answer the question from Phase 1.

### Phase 3: Data Preparation

This is usually the most time-consuming phase, accounting for roughly 70 to 80 percent of total project effort. Raw data almost never arrives in a usable state. It may have missing values, duplicate records, inconsistent formatting, or values that fall outside plausible ranges. In this phase the data is cleaned, transformed, and restructured into a format that an algorithm can process. The IBM CRISP-DM documentation describes this as covering "all activities involved in familiarizing oneself with the data."

### Phase 4: Modeling — Where Machine Learning Enters

Once the data is properly prepared, the team selects and trains a model. The algorithm learns from the prepared dataset, adjusting its parameters to capture the patterns in the training data. CRISP-DM (Studocu, 2025) describes this as a stage where "various modeling techniques depending on the problem statement are selected and applied, and their parameters are calibrated to find optimal modeling performance." Common choices include Linear Regression, Decision Trees, Random Forests, and Neural Networks.

ML belongs here specifically because it requires clean, well-formatted, and sufficiently large data to learn from. Those conditions are produced in the earlier phases. Attempting to train a model on unprepared data produces unreliable or biased results.

### Phase 5: Evaluation

The trained model is tested on data it has not seen before. The team checks whether it actually answers the original business question and whether it meets the performance thresholds set at the start. Common metrics include accuracy, precision, recall, and F1-score.

### Phase 6: Deployment

The validated model is integrated into a production environment — a recommendation engine, a fraud detection system, or a clinical decision-support tool. Monitoring is set up to track performance over time, because real-world data changes in ways the training data did not anticipate. This gradual shift is called concept drift.

---

## Q3 — Supervised vs. Unsupervised Learning

### Supervised Learning

In supervised learning, the model trains on a **labeled dataset**: each input record is already paired with a known, correct output. The algorithm learns the relationship between inputs and outputs during training, then applies that relationship to predict outputs for new inputs it has not seen.

IBM (2025) explains that "the algorithm 'learns' from the training data set by iteratively making predictions on the data and adjusting for the correct answer." Supervised learning has two main task types:

- **Classification** predicts a category (spam or not spam, disease present or absent).
- **Regression** predicts a continuous number (a house price, tomorrow's temperature).

**Example:** A spam filter trains on thousands of emails already labeled as spam or legitimate. The model identifies which words, sender patterns, and structural features correlate with spam, then applies those findings to classify emails it has never encountered.

### Unsupervised Learning

Unsupervised learning works with **unlabeled data**. There are no predefined correct answers. The algorithm must find structure, patterns, or groupings on its own.

AWS (2025) describes it this way: "unsupervised machine learning is when you give the algorithm input data without any labeled output data. Then, on its own, the algorithm identifies patterns and relationships in and between the data."

The two main unsupervised tasks are:

- **Clustering** groups similar data points together (K-Means groups customers by purchasing behavior).
- **Dimensionality Reduction** simplifies complex data into fewer variables while retaining the most important information (PCA compresses high-dimensional data for visualization).

**Example:** A retail company wants to understand its customer base but has no predefined categories. An unsupervised clustering algorithm processes purchase history data and identifies natural groupings: frequent bulk buyers, seasonal shoppers, and one-time visitors. The company did not define these groups in advance — the algorithm found them.

### Comparison

| Feature | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Data type | Labeled (inputs paired with known outputs) | Unlabeled (inputs only) |
| Human involvement | High — labels must be created manually | Lower during training, higher for interpreting results |
| Goal | Predict a known target | Discover hidden structure |
| Output | Prediction or class label | Clusters, associations, reduced features |
| Accuracy | Generally higher and verifiable | Harder to evaluate objectively |
| Common algorithms | Linear Regression, SVM, Neural Networks | K-Means, DBSCAN, PCA |
| Example applications | Spam filter, house price prediction, medical diagnosis | Customer segmentation, anomaly detection, topic modeling |

---

## Q4 — Overfitting: Causes and Prevention

### What Overfitting Is

Overfitting happens when a machine learning model learns the training data too closely — memorizing specific patterns, noise, and random variation rather than the underlying rules that would apply to new data. The result is strong performance on training data and poor performance on anything the model has not seen before.

Lightly.ai (2024) describes it this way: "instead of capturing the underlying patterns, the model memorizes the data, leading to poor generalization." The most visible sign is a large gap between training accuracy and test accuracy.

### The Bias-Variance Tradeoff

Overfitting sits within a broader concept called the **bias-variance tradeoff**:

- **Bias** is the error from assumptions that are too simple. The model misses real patterns in the data. This is underfitting.
- **Variance** is the error from being too sensitive to the training data. The model fits noise rather than signal. This is overfitting.

Aya Data (2025) states: "High Variance (Overfitting): The model is complex and highly sensitive to the training data, leading to low training error but high test error."

The goal is a model with low enough bias to learn genuine patterns and low enough variance to apply them correctly to new data.

### Causes

| Cause | Explanation |
|---|---|
| Model is too complex | Too many parameters relative to the amount of data (e.g., a deep neural network on a small dataset) |
| Insufficient training data | Too few examples for the model to separate genuine patterns from noise |
| Too many training iterations | The model has too many opportunities to memorize individual examples |
| Noisy or irrelevant features | Irrelevant variables lead the model to learn correlations that do not hold generally |
| No regularization | Without constraints, the model grows as complex as needed to fit the training data exactly |

### Prevention Techniques

**1. More training data.**
Larger datasets make individual examples harder to memorize and push the model toward general patterns.

**2. Regularization (L1 and L2).**
A penalty is added to the loss function to discourage large parameter values. L1 (Lasso) can push some weights to zero, effectively removing features. L2 (Ridge) shrinks all weights proportionally without zeroing them. Jaro Education (2025) notes that "increasing regularisation strength lowers variance by preventing overfitting."

**3. Cross-validation.**
K-fold cross-validation divides the data into k subsets, trains on k-1 of them, and tests on the remaining one, rotating through all subsets. The resulting performance estimate is more reliable than a single holdout evaluation.

**4. Early stopping.**
Training is halted when performance on a validation set stops improving, before the model has time to memorize training examples.

**5. Simpler models.**
If a simpler model achieves comparable performance, it is preferable. Complexity that is not justified by the data creates room for overfitting.

**6. Dropout (neural networks).**
During training, neurons are randomly deactivated at each step. This prevents the network from relying heavily on any single neuron or pathway, producing more robust representations.

---

## Q5 — Training and Test Data Split

### The Three Subsets

Before training begins, the full dataset is divided into separate subsets so that the model is evaluated on data it has genuinely never seen. The standard subsets are:

- **Training Set.** The data the model learns from. Parameters are adjusted based entirely on this portion.
- **Test Set.** Data the model never sees during training. It is used once, at the end, to measure real-world performance.
- **Validation Set** (used in many projects). Checked during development to tune hyperparameters without contaminating the test set.

### Common Split Ratios

| Split | Training | Validation | Test |
|---|---|---|---|
| Simple 80/20 | 80% | — | 20% |
| Three-way 70/15/15 | 70% | 15% | 15% |
| Large-data 80/10/10 | 80% | 10% | 10% |

A published study in PMC (National Institutes of Health, 2024) tested ratios from 60:40 to 95:05 on medical imaging data and found "significant variations in accuracies across these ratios, emphasizing the critical need to strike a balance to avoid overfitting or underfitting." The right ratio depends on dataset size and the task — there is no universal answer.

**Randomization before splitting** is essential. If the data has an inherent order (time, collection sequence, alphabetical order), taking the first 80 percent as training data produces a biased split. Shuffling before splitting ensures that both sets represent the overall distribution.

**Stratified sampling** is used when class proportions are uneven. If only 5 percent of records represent fraud, a random split might give the test set 1 percent fraud cases or 10 percent. Stratified sampling preserves the original proportion in both sets.

### Why the Split Is Necessary

A model evaluated on its own training data will almost always appear to perform well, even if it has only memorized examples rather than learned anything transferable. This is the definition of overfitting.

A separate test set provides an honest measure of how the model will behave after deployment, when it encounters data it has never processed before. Encord (2026) puts it directly: "splitting data helps prevent overfitting and ensures the model generalizes well... this creates a more robust and versatile model for real-world use."

The test set represents the future. It shows whether the model learned something real or just memorized the past.

---

## Q6 — Case Study: ML in Healthcare

**Source:** Iparraguirre-Villanueva, O., Espinola-Linares, K., Flores Castaneda, R. O., and Cabanillas-Carbonell, M. (2023). Application of Machine Learning Models for Early Detection and Accurate Classification of Type 2 Diabetes. *Diagnostics*, 13(14), 2383. https://doi.org/10.3390/diagnostics13142383

### Background

Type 2 diabetes is one of the most common chronic diseases worldwide. By 2021, approximately 537 million adults were living with diabetes. The International Diabetes Federation projected that number would reach 643 million by 2030 and 784 million by 2045. Detecting diabetes early matters because the complications that develop over time — kidney failure, nerve damage, retinal disease — are substantially harder to manage than the disease itself at an early stage.

Standard diagnosis depends on laboratory tests and clinician judgment. This study asked whether supervised machine learning could improve or support early detection using routine clinical data.

### What the Researchers Did

The researchers used the **Pima Indian Diabetes Dataset** (768 patient records) with eight input variables: number of pregnancies, blood glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. Each record was labeled diabetic (1) or non-diabetic (0), making this a binary classification task.

Five supervised ML algorithms were trained and compared:

| # | Algorithm |
|---|---|
| 1 | K-Nearest Neighbors (K-NN) |
| 2 | Bernoulli Naive Bayes (BNB) |
| 3 | Decision Tree (DT) |
| 4 | Logistic Regression (LR) |
| 5 | Support Vector Machine (SVM) |

Each model was evaluated using accuracy, precision, recall, and F1-score on a held-out test set.

### Key Findings

All five models produced meaningful classification results from routine clinical data. The Support Vector Machine achieved the highest accuracy overall. The study concluded that ML models trained on a patient's medical history and risk factors can reliably predict diabetes risk and could function as a clinical decision-support tool.

### CRISP-DM Coverage

| Lifecycle Phase | What the Study Did |
|---|---|
| Business Understanding | Defined the goal: classify Type 2 diabetes risk early to prevent complications |
| Data Understanding | Used 768 records with 8 clinical features from the Pima Indian Diabetes Dataset |
| Data Preparation | Handled missing values, normalized features, and prepared data for each algorithm |
| Modeling | Trained 5 supervised ML classifiers (K-NN, BNB, DT, LR, SVM) |
| Evaluation | Compared models using accuracy, precision, recall, and F1-score |
| Deployment (proposed) | Recommended the best-performing model for use as a clinical support tool |

The study covers the full CRISP-DM cycle. The data pipeline work — collection, cleaning, feature preparation — is Data Science. The classifiers (SVM, Decision Tree, and the rest) are the Machine Learning component embedded within that process.

---

## References

1. Sartorius. (2020). *Understanding the relationship between Data Science, Artificial Intelligence and Machine Learning*. https://www.sartorius.com/en/knowledge/science-snippets/data-science-vs-artificial-intelligence-vs-machine-learning-602514

2. Pangeanic. (2023). *The relationship between data science and machine learning*. https://blog.pangeanic.com/trelationship-between-data-science-and-machine-learning

3. IBM. (2025). *Data science vs. machine learning: What's the difference?* https://www.ibm.com/think/topics/data-science-vs-machine-learning

4. Syracuse University iSchool. (2025). *Data Science vs. Machine Learning: Key differences explained*. https://ischool.syracuse.edu/data-science-vs-machine-learning/

5. Excelsior University. (n.d.). *Chapter 1.4: The Data Science Lifecycle (CRISP-DM)*. https://express.excelsior.edu/datascience/chapter/1-4-the-data-science-lifecycle-crisp-dm/

6. Melo, G. (2024). *CRISP-DM: A comprehensive guide to the structured Data Science methodology*. Medium. https://medium.com/@gcesarmelo7/crisp-dm

7. Studocu. (2025). *CRISP-DM Model Lifecycle: Steps, challenges, and best practices*. https://www.studocu.com

8. IBM. (2025). *Supervised vs. unsupervised learning: What's the difference?* https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning

9. AWS. (2025). *Supervised vs unsupervised learning: Difference between machine learning algorithms*. https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/

10. Lightly.ai. (2024). *Overfitting in machine learning: Causes, detection, and prevention*. https://www.lightly.ai/blog/overfitting

11. Aya Data. (2025). *Underfitting vs. overfitting in machine learning: A complete 2025 guide*. https://www.ayadata.ai/a-guide-to-overfitting-and-underfitting-in-machine-learning/

12. Jaro Education. (2025). *Bias-variance tradeoff in machine learning explained with examples*. https://www.jaroeducation.com/blog/bias-variance-tradeoff-in-machine-learning

13. EMB Global. (2024). *Train and test data: Best practices for machine learning*. https://blog.emb.global/train-and-test-data/

14. Encord. (2026). *How to split machine learning datasets: Training, validation and test sets*. https://encord.com/blog/train-val-test-split/

15. Alnuaimi, N., Dermawan, D., Kamaruzaman, H. F., and Alotaiq, N. (2024). *Trade-off between training and testing ratio in machine learning for medical image processing*. PMC / National Institutes of Health. https://pmc.ncbi.nlm.nih.gov/articles/PMC11419616/

16. Iparraguirre-Villanueva, O., Espinola-Linares, K., Flores Castaneda, R. O., and Cabanillas-Carbonell, M. (2023). Application of machine learning models for early detection and accurate classification of type 2 diabetes. *Diagnostics*, 13(14), 2383. https://doi.org/10.3390/diagnostics13142383

---


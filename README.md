#  Traditional Natural Language Processing (NLP) for Wine Classification  
## Distinct Algorithms for Predictive Modeling  
#### Logistic Regression  
#### Random Forest  
#### and XGBoost  

---

##  Overview
This project demonstrates a **traditional NLP text classification pipeline** for predicting wine **varieties and color categories (red, white, rosé)** based on natural-language reviews.  
It showcases a complete end-to-end classical NLP workflow — data cleaning, vectorization (TF-IDF/Bag-of-Words), and machine-learning models such as **Logistic Regression, Random Forest, and XGBoost**.

---

##  Dataset
- **Source:** Kaggle — *Wine Reviews Dataset (Winemag data, ~150k samples)*  
- **Samples after cleaning:** ≈ 81,000 reviews  
- **Features:** `country`, `description`, `price`, `province`, `region_1`, `variety`, `winery`  
- **Target labels:** Wine `colour` (red / white / rosé) and `variety` (619 unique classes)  
- **[Wine Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/zynicide/wine-reviews)

### Data Cleaning Steps
- Removed nulls, duplicates, and non-informative entries  
- Lowercased all text and normalized Unicode accents  
- Mapped each wine variety to a general color label  
- Standardized categorical and numerical features  

---

## Preprocessing Pipeline
1. **Text Cleaning**
   - Lowercasing, punctuation & digit removal  
   - Accent normalization using `unicodedata.normalize()`  
   - Regex filtering for special symbols  

2. **Tokenization & Stopword Removal**
   - Used NLTK tokenizer and extended English stopword list  

3. **Lemmatization**
   - Applied `WordNetLemmatizer` to normalize word forms  

4. **Feature Engineering**
   - Added binary color columns (`red`, `white`, `rosé`)  
   - Created cleaned and normalized text columns (`description_clean`, `description_text`)  

5. **Vectorization**
   - `TfidfVectorizer(ngram_range=(1, 2))`  
   - Combined sparse TF-IDF matrix with categorical encodings  

---

##  Modeling
| Model | Technique | Key Parameters | Accuracy |
|:------|:-----------|:---------------|:---------:|
| **Logistic Regression** | Linear baseline | `C=2`, L2 penalty | 0.582 |
| **Random Forest** | Ensemble bagging | 200 trees, `max_depth=15` | 0.567 |
| **XGBoost** | Gradient boosting | 500 trees, `eta=0.1`, `max_depth=10` | **0.635 ✅** |

- **Train/Test Split:** 80/20 (64k / 16k)  
- **Cross-Validation:** 3-fold `GridSearchCV` for tuning  
- **Evaluation:** Accuracy, Precision, Recall, F1-Score  

---

##  Evaluation Metrics
| Metric | Logistic Regression | Random Forest | XGBoost |
|:--------|:------------------:|:-------------:|:-------:|
| **Accuracy** | 0.582 | 0.567 | **0.635** |
| **Macro F1** | 0.50 | 0.47 | **0.56** |
| **Weighted F1** | 0.60 | 0.58 | **0.64** |

> The **XGBoost model** achieved the best overall performance, especially on high-frequency varieties like *Pinot Noir* and *Cabernet Sauvignon.*

---

##  Results & Analysis
- **Top Terms:** "oak", "acid", "ripe fruit", "fresh finish" strongly influenced color and variety prediction.  
- **Model Insights:**  
  - Logistic Regression captured frequent n-gram correlations.  
  - Random Forest performed better on categorical splits.  
  - XGBoost handled rare varieties and non-linear relationships most effectively.  
- **Error Analysis:** Confusion between similar red wine types (e.g., Syrah vs. Grenache).

---

##  How to Run

###  Environment Setup
```bash
git clone https://github.com/vishalgwu/Traditional-NLP-for-wine.git
cd Traditional-NLP-for-wine
conda create -n nlp-wine python=3.10


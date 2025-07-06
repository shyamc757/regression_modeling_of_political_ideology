# 🗳️ Regression Modeling of Political Ideology using Tweets

This project analyzes political discourse on Twitter by U.S. Congress members (2008–2020) and uses machine learning to predict their ideological positions. Leveraging DW-NOMINATE scores and modern NLP techniques, we model how tweet content reflects political leanings on the liberal–conservative spectrum.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🧠 Project Overview

- **Objective**: Predict two DW-NOMINATE ideology dimensions from tweet content.
- **Data**: 333K+ tweets for training and 135K+ tweets for testing.
- **Techniques**: DistilBERT embeddings, Random Forest regression, topic modeling (NMF, LDA).
- **Result**: Achieved RMSE of **0.277**, outperforming the benchmark of 0.36.

## 📂 Repository Structure

```
regression_modeling_of_political_ideology/
│
├── descriptive.ipynb         # EDA notebook (hashtag analysis, DW-NOMINATE insights)
├── kaggle_project_report.pdf # Full project report
├── lemmatizer.py             # Custom text lemmatization/preprocessing script
├── main.ipynb                # Main notebook for modeling and training
├── README.md                 # Project overview (this file)
```

## 📊 Key Dataset Features

- `full_text`: Tweet content  
- `hashtags`: Associated hashtags  
- `favorite_count`, `retweet_count`: Tweet engagement  
- `year`: Year posted  
- `ideology_dim1`, `ideology_dim2`: DW-NOMINATE ideology scores (targets)

## 🔍 Analysis Highlights

- Top hashtags: #COVID19, #Obamacare, #tcot, etc.
- Grouped tweets into 4 ideological bins and analyzed linguistic trends
- Ridge plots show ideological divergence over time
- Topic modeling with **NMF** and **LDA** for deeper text insights
- Calculated most ideologically distant tweet pairs using Euclidean distances

## 🛠️ Modeling Pipeline

### Preprocessing
- Cleaned text: removed URLs, emojis, punctuation, stopwords, etc.
- Tokenized using `DistilBERT` and converted tweets to embeddings

### Modeling
- Tested: KNN, XGBoost, Random Forest
- Final: Random Forest + MultiOutputRegressor with GridSearch
- Metric: RMSE (Root Mean Square Error)

| Model              | RMSE        |
|-------------------|-------------|
| XGBoost           | ~0.35       |
| KNN               | ~0.36       |
| **Random Forest** | **0.2771**  |

## 🧪 Topic Modeling

| Model | Summary |
|-------|---------|
| **NMF** | More specific, focused topics via TF-IDF |
| **LDA** | Broader, general themes via raw term counts |

Each model contributed complementary insight into the themes driving political messaging.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 👥 Authors

- [Shyam Shah](https://github.com/shyamc757)
- [Neel Agarwal](https://github.com/neelagarwal98/dw-nominate-scores)

## 📫 Contact

Feel free to reach out for collaboration or feedback via GitHub or LinkedIn.

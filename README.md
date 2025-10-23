# Prediction of Figure Skating Element Execution

**Client:** [GoProtect Insurance Company](https://www.goprotect.ru/)  
**Role:** Data Scientist  
**Duration**: 2 months

## Project Overview

The project was developed as part of *“Мой Чемпион” (“My Champion”)* — a service helping figure skating schools and coaches monitor athletes’ performance and plan training programs.  
This project explored the feasibility of predicting figure skating element performance using machine learning on real-world competition data.

**Goal:**
Develop a machine learning model to predict which figure skating elements an athlete is likely to successfully perform during competitions, based on their previous results and performance history.

## Tech Stack
Python 3.10+  
Core Libraries:
- `pandas`, `numpy` – data manipulation and preprocessing
- `matplotlib`, `seaborn` – data visualization  

Machine Learning & Modeling:  
- `catboost` – gradient boosting models (CatBoostClassifier, CatBoostRegressor)
- `scikit-learn` – metrics, preprocessing, model selection
- `phik` – advanced correlation analysis and visualization

## Data

Historical data on single skaters’ competition results from multiple schools, including:

* Competition details (date, location, program type)
* Element-level results and errors
* Base scores, GOE (Grade of Execution) and total score

**Dataset summary**:
- 3 competition seasons
- 4596 athletes from 239 schools
- 142 competitions in 3 locations

Additional domain references:

* [Figure Skating Elements List (2023/24)](https://eislauf-union.de/files/users/997/Elemente-Liste2023_24.pdf)
* [Base Element Scores](https://fsrussia.ru/files/docs/SSPScomm_2475.pdf)
* [Judging guidelines (Technical Handbook)](https://fsrussia.ru/files/docs/tp_handbook_singles_2324.pdf)

## Methodology

**Main steps:**

1. **Data preprocessing & EDA**

   * Handling missing values, inconsistencies, duplicates. Encoding
   * Feature engineering and merging hierarchical data (athlete → tournament → program → element)

2. **Feature selection**

   * Element type, program type, tournament frequency, previous GOE, element order, etc.

3. **Modeling**

   * Multi-label classification (CatBoostClassifier) for element execution prediction
   * Regression (CatBoostRegressor) for GOE prediction

4. **Evaluation**

   * Comparison with constant (dummy) baselines
   * Feature importance analysis

**Features Used**
1. Competition category
2. School (missing values filled with 0)
3. Tournament number in season
4. Days since last tournament & tournament duration
5. Competition location
6. Program type and placement in previous tournament
7. Element type (jump, spin, step)
8. Whether element was part of a combination
9. Element order in the program
9. Element name (code)
10. Element level (rotations or spin level)
11. Previous GOE and average score

**Target Variables**
1. Multi-label classification: element execution errors
* V – rotation error
* q, ur, uur – under-rotation (¼, 90°-180°, >180°)
* ue, e – wrong or unclear edge
* F – fall (including combination/sequence falls)
2. Regression
* GOE (Grade of Execution)
  
## Results

| Task             | Model              | Metric   | Score  | Dummy  |
| ---------------- | ------------------ | -------- | ------ | ------ |
| Error prediction | CatBoostClassifier | Accuracy | 0.7605 | 0.6185 |
|                  |                    | F1 Score | 0.2043 | 0.2939 |
| GOE prediction   | CatBoostRegressor  | RMSE     | 0.7698 | 0.7922 |

**Most important features:**

* Element name and level
* Presence of fall
* Previous GOE value
* Under-rotation flags

**Conclusion:**
* The baseline model achieved only a slight improvement over dummy predictions, showing strong data imbalance and limited correlation between available features and performance outcomes.
* Strongest correlations were found between GOE and features such as falls, element level, and previous GOE.

## Key Insights & Recommendations

### Challenges:

* Highly imbalanced data (few errors vs. many correct executions)
* Complex hierarchical structure of the dataset
* Limited and inconsistent domain annotations
* Missing information about programs, schools, and tournament structure
* Variability between athletes (skill level, frequency of participation)

To address these, the dataset was filtered to:
- Include only the latest season
- Keep only athletes with more than one tournament
- Exclude rarely occurring or trivial elements (e.g., single jumps, certain spins)

### Recommendations:
**For modeling:**
* Balance classes or aggregate error types
* Add more contextual features and domain data
* Add historical GOE/error sequences as features
* Try advanced models (e.g., LSTM for temporal dependencies)
* Focus on elements with high error rates

**For the client:**
* Enrich dataset with more athlete and competition details
* Include more tournaments, programs, and seasons

## Notes

This project was developed as part of an MVP/research initiative for GoProtect Insurance.
The analysis and models are for research purposes only and do not represent production-ready results.

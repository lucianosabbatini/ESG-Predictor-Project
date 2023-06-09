# ESG Predictor

# Ironhack Data Analytics Bootcamp Final Project

**Luciano Sabbatini**  
**IronHack, Germany 21 Apr 2023**

## Goals

* To evaluate the efficiency of a model and application to predict ESG Risk Scores based on financial datapoints and using Sustainalytics methodology as reference metrics.
* Reference: https://www.sustainalytics.com/corporate-solutions/esg-solutions/esg-risk-ratings#esg 

Used:

 * Python
 * Statistical Analysis
 * Data Visualization
 * Jupyter Notebook
 * Tableau
 * Machine Learning (Randon Forest)
 * Web Scrapping and APIs
  
  ## Data Sources:

### Overview: 
* First Dataset is about the mean salaries of workers on Russel 3000 companies:
	* Dataset source: https://www.kaggle.com/datasets/salimwid/latest-top-3000-companies-ceo-salary-202223
* Following data, and the core of the model learning algorythims, all built from web scrapping and APIs:
  * API from U.S. Security Exchange Commission (SEC) : https://www.sec.gov/files/company_tickers_exchange.json
  * API from U.S. Security Exchange Commission (SEC): https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  * Web-Scrapping Wikipedia: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
  * Web-Scrapping Yahoo Finances: https://finance.yahoo.com/quote/{ticker}/sustainability?p={ticker} / https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}
  * Web-Scrapping from Ameritrade: https://research.tdameritrade.com/grid/public/research/stocks/summary?fromPage=overview&display=&fromSearch=true&symbol={ticker}

 
### Data Wrangling and Cleaning
  
- Dropping rows of nulls, filling nulls
- Deleting  columns with identifier id on it 
- Converting UNIX time into datetime
- Analysing unique features of the dataset 
- Analysing the relations between the financial factors and overall ratings
- Creating plots for better visualization
- Creating new columns
- Creating step conclusions for each part of the analysis

### EDA and Visualization
- Used Python and Tableau to visualize my overall data.
- Used hyperparamether and feature selection techniques to find the most relevant features for the model. 

### Model Training and Evaluation
- Models : LinearRegression, MLPRegressor, KNN neighbors, RandonForestRegressor (the selected one)
- Compared accuracy and cross validation 

### Presentation and Utilization
- Powerpoint and a Streamlit app for providing a free ESG Calculator tool for companies.

## Outlook

- Our model is a prototype based on Sustainalytics metrics, which include more than 80 non-financial datapoints, and we used only financial datapoints. Therefore, our model results can be considered a good foundantion for building a truly useful model. 

### Contributors

- Please feel free to contribute to this project by submitting feedback or pull requests. All contributions are welcome!

- Do not hesitate to reach out to me at https://www.linkedin.com/in/luciano-sabbatini/.


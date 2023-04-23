# ESG Predictor

# Ironhack Data Analytics Bootcamp Final Project

**Luciano Sabbatini**  
**IronHack, Germany 21 Apr 2023**

## Goals

* To create a model and application to predict ESG Scores. 

Used:

	* Python
  * Statistical analysis
	* Data visualization
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
- Analysing the relations between the sensorial factors and overall ratings ratings
- Creating plots for better visualization
- Creating new columns
- Creating Step conclusions for each part of the analysis
- Building a recommendation system
- Testing the model.

### Data Exploration and Visualization
- Used Python and Tableau to visualize my overall data.

### Model Training and Evaluation
- Define predictors and target values (X, y)
- Standard scaling for numericals : for Train and Test set
- Models : LinearRegression, MLPRegressor, KNN neighbors, RandonForestRegressor (the selected one)
- Compared accuracy and cross validation 

### Presentation and Utilization
- Created a Streamlit app for providing a free ESG Calculator tool for companies.

## Outlook

- Our model is a prototype due the lack of more environment and social diverse data at the given time for the project. Once this data is added as feature for the model, it would improve significantly and make it very useful.


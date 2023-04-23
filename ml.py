# Import python libraries
from pickle import load
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

def ml_environment():
    
    st.title("Enviroment Score")
    model = load(open('model.pkl', 'rb'))
    scaler = load(open('scaler.pkl', 'rb'))
    encoder = load(open('encoder.pkl', 'rb'))


# Define the Streamlit app
    st.title("")
    valid_industries = ['Communication Services', 'Consumer_Discretionary', 'Consumer Staples',
                                             'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology',
                                             'Materials', 'Real Estate', 'Utilities']

# Ask for user input
    industry = st.sidebar.selectbox("Select your segment", valid_industries, key = '10')
    entity_public_float = st.number_input('Public Float',min_value=0)
    comprehensive_loss_net_of_tax = st.number_input('Comprehensive Loss Net of Tax',-1000000000000.0, 100000000.0, 0.0, 1.0)
    assets = st.number_input('Assets',min_value=0)
    cash_equivalents = st.number_input('Cash Equivalents',-1000000000000.0, 100000000.0, 0.0, 1.0)
    comprehensive_income_net_of_tax = st.number_input('Comprehensive Income Net of Tax',min_value=0)
    earnings_per_share_diluted = st.number_input('Earnings per Share Diluted',min_value=0)
    income_tax_expense_benefit = st.number_input('Income Tax Expense Benefit',min_value=0)
    liabilities_and_stockholders_equity = st.number_input('Liabilities and Stockholders Equity',min_value=0)
    net_from_financing_activities = st.number_input('Net from Financing Activities',-1000000000000.0, 100000000.0, 0.0, 1.0)
    investments = st.number_input('Investments',-1000000000000000.0, 100000000.0, 0.0, 1.0)
    net_from_operating_activities = st.number_input('Net from Operating Activities',min_value=0)
    net_income_loss = st.number_input('Net Income Loss',min_value=0)
    operating_lease_liability = st.number_input('Operating Lease Liability',min_value=0)
    retained_earnings = st.number_input('Retained Earnings',min_value=0)
    stockholders_equity = st.number_input('Stockholders Equity',min_value=0)
    diluted_shares_outstanding = st.number_input('Diluted Shares Outstanding',min_value=0)
    median_worker_pay = st.number_input('Median Worker Pay',min_value=0)
    ceo_salary = st.number_input('CEO Salary',min_value=0)
    revenue = st.number_input('Revenue',min_value=0)
    gross_profit = st.number_input('Gross Profit',min_value=0)
    total_debt = st.number_input('Total Debt',min_value=0)
    market_cap = st.number_input('Market Cap',min_value=0)

# Preprocess the user input
    if st.button("Get Your Prediction"):
        X = pd.DataFrame({
        'industry': [industry],
        'entity_public_float': [entity_public_float],
        'comprehensive_loss_net_of_tax': [comprehensive_loss_net_of_tax],
        'assets': [assets],
        'cash_equivalents': [cash_equivalents],
        'comprehensive_income_net_of_tax': [comprehensive_income_net_of_tax],
        'earnings_per_share_diluted': [earnings_per_share_diluted],
        'income_tax_expense_benefit': [income_tax_expense_benefit],
        'liabilities_and_stockholders_equity': [liabilities_and_stockholders_equity],
        'net_from_financing_activities': [net_from_financing_activities],
        'investments': [investments],
        'net_from_operating_activities': [net_from_operating_activities],
        'net_income_loss': [net_income_loss],
        'operating_lease_liability': [operating_lease_liability],
        'retained_earnings': [retained_earnings],
        'stockholders_equity': [stockholders_equity],
        'diluted_shares_outstanding': [diluted_shares_outstanding],
        'median_worker_pay': [median_worker_pay],
        'ceo_salary': [ceo_salary],
        'revenue': [revenue],
        'gross_profit': [gross_profit],
        'total_debt': [total_debt],
        'market_cap': [market_cap]})

        X = X
        numerical = X.select_dtypes(include = np.number)
        categorical = X.select_dtypes(include = np.object)

        cat_transformed = encoder.transform(categorical)
        col_names = ['industry_Communication Services', 'industry_Consumer Discretionary',
       'industry_Consumer Staples', 'industry_Energy', 'industry_Financials',
       'industry_Health Care', 'industry_Industrials',
       'industry_Information Technology', 'industry_Materials',
       'industry_Real Estate', 'industry_Utilities']
        
        categorical = pd.DataFrame(cat_transformed.toarray(), columns = col_names)

        # Joning dataframes
        X = pd.concat([numerical, categorical], axis=1)
        # Scaling data
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

        # Making predictions
        prediction = model.predict(X_scaled_df)
        return prediction

def ml_social():
    
    st.title("Social Score")
    model = load(open('model2.pkl', 'rb'))
    scaler = load(open('scaler2.pkl', 'rb'))
    encoder = load(open('encoder2.pkl', 'rb'))


# Define the Streamlit app
    st.title("")
    valid_industries = ['Communication Services', 'Consumer_Discretionary', 'Consumer Staples',
                                             'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology',
                                             'Materials', 'Real Estate', 'Utilities']

# Ask for user input
    industry = st.sidebar.selectbox("Select your segment", valid_industries, key = '10')
    entity_public_float = st.number_input('Public Float',min_value=0)
    comprehensive_loss_net_of_tax = st.number_input('Comprehensive Loss Net of Tax',-10000000000000.0, 100000000.0, 0.0, 1.0)
    assets = st.number_input('Assets',min_value=0)
    cash_equivalents = st.number_input('Cash Equivalents',-10000000000000.0, 100000000.0, 0.0, 1.0)
    comprehensive_income_net_of_tax = st.number_input('Comprehensive Income Net of Tax',min_value=0)
    earnings_per_share_diluted = st.number_input('Earnings per Share Diluted',min_value=0)
    income_tax_expense_benefit = st.number_input('Income Tax Expense Benefit',min_value=0)
    liabilities_and_stockholders_equity = st.number_input('Liabilities and Stockholders Equity',min_value=0)
    net_from_financing_activities = st.number_input('Net from Financing Activities',-10000000000000.0, 100000000.0, 0.0, 1.0)
    investments = st.number_input('Investments',-1000000000000000.0, 100000000.0, 0.0, 1.0)
    net_from_operating_activities = st.number_input('Net from Operating Activities',min_value=0)
    net_income_loss = st.number_input('Net Income Loss',min_value=0)
    operating_lease_liability = st.number_input('Operating Lease Liability',min_value=0)
    retained_earnings = st.number_input('Retained Earnings',min_value=0)
    stockholders_equity = st.number_input('Stockholders Equity',min_value=0)
    diluted_shares_outstanding = st.number_input('Diluted Shares Outstanding',min_value=0)
    median_worker_pay = st.number_input('Median Worker Pay',min_value=0)
    ceo_salary = st.number_input('CEO Salary',min_value=0)
    revenue = st.number_input('Revenue',min_value=0)
    gross_profit = st.number_input('Gross Profit',min_value=0)
    total_debt = st.number_input('Total Debt',min_value=0)
    market_cap = st.number_input('Market Cap',min_value=0)

# Preprocess the user input
    if st.button("Get Your Prediction"):
        X = pd.DataFrame({
        'industry': [industry],
        'entity_public_float': [entity_public_float],
        'comprehensive_loss_net_of_tax': [comprehensive_loss_net_of_tax],
        'assets': [assets],
        'cash_equivalents': [cash_equivalents],
        'comprehensive_income_net_of_tax': [comprehensive_income_net_of_tax],
        'earnings_per_share_diluted': [earnings_per_share_diluted],
        'income_tax_expense_benefit': [income_tax_expense_benefit],
        'liabilities_and_stockholders_equity': [liabilities_and_stockholders_equity],
        'net_from_financing_activities': [net_from_financing_activities],
        'investments': [investments],
        'net_from_operating_activities': [net_from_operating_activities],
        'net_income_loss': [net_income_loss],
        'operating_lease_liability': [operating_lease_liability],
        'retained_earnings': [retained_earnings],
        'stockholders_equity': [stockholders_equity],
        'diluted_shares_outstanding': [diluted_shares_outstanding],
        'median_worker_pay': [median_worker_pay],
        'ceo_salary': [ceo_salary],
        'revenue': [revenue],
        'gross_profit': [gross_profit],
        'total_debt': [total_debt],
        'market_cap': [market_cap]})

        X = X
        numerical = X.select_dtypes(include = np.number)
        categorical = X.select_dtypes(include = np.object)

        cat_transformed = encoder.transform(categorical)
        col_names = ['industry_Communication Services', 'industry_Consumer Discretionary',
       'industry_Consumer Staples', 'industry_Energy', 'industry_Financials',
       'industry_Health Care', 'industry_Industrials',
       'industry_Information Technology', 'industry_Materials',
       'industry_Real Estate', 'industry_Utilities']
        
        categorical = pd.DataFrame(cat_transformed.toarray(), columns = col_names)

        # Joning dataframes
        X = pd.concat([numerical, categorical], axis=1)
        # Scaling data
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

        # Making predictions
        prediction2 = model.predict(X_scaled_df)
        return prediction2

def ml_governance():
    
    st.title("Governance Score")
    model = load(open('model3.pkl', 'rb'))
    scaler = load(open('scaler3.pkl', 'rb'))
    encoder = load(open('encoder3.pkl', 'rb'))


# Define the Streamlit app
    st.title("")
    valid_industries = ['Communication Services', 'Consumer_Discretionary', 'Consumer Staples',
                                             'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology',
                                             'Materials', 'Real Estate', 'Utilities']
    default_comprehensive_loss_net_of_tax=361000000

# Ask for user input
    industry = st.sidebar.selectbox("Select your segment", valid_industries, key = '10')
    entity_public_float = st.number_input('Public Float',min_value=0)
    comprehensive_loss_net_of_tax = st.number_input('Comprehensive Loss Net of Tax', -10000000000000.0, 100000000.0, 0.0, 1.0)
    assets = st.number_input('Assets',min_value=0)
    cash_equivalents = st.number_input('Cash Equivalents',-10000000000000.0, 100000000.0, 0.0, 1.0)
    comprehensive_income_net_of_tax = st.number_input('Comprehensive Income Net of Tax',min_value=0)
    earnings_per_share_diluted = st.number_input('Earnings per Share Diluted',min_value=0)
    income_tax_expense_benefit = st.number_input('Income Tax Expense Benefit',min_value=0)
    liabilities_and_stockholders_equity = st.number_input('Liabilities and Stockholders Equity',min_value=0)
    net_from_financing_activities = st.number_input('Net from Financing Activities',-10000000000000.0, 100000000.0, 0.0, 1.0)
    investments = st.number_input('Investments',-1000000000000000.0, 100000000.0, 0.0, 1.0)
    net_from_operating_activities = st.number_input('Net from Operating Activities',min_value=0)
    net_income_loss = st.number_input('Net Income Loss',min_value=0)
    operating_lease_liability = st.number_input('Operating Lease Liability',min_value=0)
    retained_earnings = st.number_input('Retained Earnings',min_value=0)
    stockholders_equity = st.number_input('Stockholders Equity',min_value=0)
    diluted_shares_outstanding = st.number_input('Diluted Shares Outstanding',min_value=0)
    median_worker_pay = st.number_input('Median Worker Pay',min_value=0)
    ceo_salary = st.number_input('CEO Salary',min_value=0)
    revenue = st.number_input('Revenue',min_value=0)
    gross_profit = st.number_input('Gross Profit',min_value=0)
    total_debt = st.number_input('Total Debt',min_value=0)
    market_cap = st.number_input('Market Cap',min_value=0)

# Preprocess the user input
    if st.button("Get Your Prediction"):
        X = pd.DataFrame({
        'industry': [industry],
        'entity_public_float': [entity_public_float],
        'comprehensive_loss_net_of_tax': [comprehensive_loss_net_of_tax],
        'assets': [assets],
        'cash_equivalents': [cash_equivalents],
        'comprehensive_income_net_of_tax': [comprehensive_income_net_of_tax],
        'earnings_per_share_diluted': [earnings_per_share_diluted],
        'income_tax_expense_benefit': [income_tax_expense_benefit],
        'liabilities_and_stockholders_equity': [liabilities_and_stockholders_equity],
        'net_from_financing_activities': [net_from_financing_activities],
        'investments': [investments],
        'net_from_operating_activities': [net_from_operating_activities],
        'net_income_loss': [net_income_loss],
        'operating_lease_liability': [operating_lease_liability],
        'retained_earnings': [retained_earnings],
        'stockholders_equity': [stockholders_equity],
        'diluted_shares_outstanding': [diluted_shares_outstanding],
        'median_worker_pay': [median_worker_pay],
        'ceo_salary': [ceo_salary],
        'revenue': [revenue],
        'gross_profit': [gross_profit],
        'total_debt': [total_debt],
        'market_cap': [market_cap]})

        X = X
        numerical = X.select_dtypes(include = np.number)
        categorical = X.select_dtypes(include = np.object)

        cat_transformed = encoder.transform(categorical)
        col_names = ['industry_Communication Services', 'industry_Consumer Discretionary',
       'industry_Consumer Staples', 'industry_Energy', 'industry_Financials',
       'industry_Health Care', 'industry_Industrials',
       'industry_Information Technology', 'industry_Materials',
       'industry_Real Estate', 'industry_Utilities']
        
        categorical = pd.DataFrame(cat_transformed.toarray(), columns = col_names)

        # Joning dataframes
        X = pd.concat([numerical, categorical], axis=1)
        # Scaling data
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

        # Making predictions
        prediction3 = model.predict(X_scaled_df)
        return prediction3

def ml_esg():
    
    st.title("ESG Overall Score")
    model = load(open('model4.pkl', 'rb'))
    scaler = load(open('scaler4.pkl', 'rb'))
    encoder = load(open('encoder4.pkl', 'rb'))


# Define the Streamlit app
    st.title("")
    valid_industries = ['Communication Services', 'Consumer_Discretionary', 'Consumer Staples',
                                             'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology',
                                             'Materials', 'Real Estate', 'Utilities']

# Ask for user input
    industry = st.sidebar.selectbox("Select your segment", valid_industries, key = '10')
    entity_public_float = st.number_input('Public Float',min_value=0)
    comprehensive_loss_net_of_tax = st.number_input('Comprehensive Loss Net of Tax',-10000000000000.0, 100000000.0, 0.0, 1.0)
    assets = st.number_input('Assets',min_value=0)
    cash_equivalents = st.number_input('Cash Equivalents',-10000000000000.0, 100000000.0, 0.0, 1.0)
    comprehensive_income_net_of_tax = st.number_input('Comprehensive Income Net of Tax',min_value=0)
    earnings_per_share_diluted = st.number_input('Earnings per Share Diluted',min_value=0)
    income_tax_expense_benefit = st.number_input('Income Tax Expense Benefit',min_value=0)
    liabilities_and_stockholders_equity = st.number_input('Liabilities and Stockholders Equity',min_value=0)
    net_from_financing_activities = st.number_input('Net from Financing Activities',-10000000000000.0, 100000000.0, 0.0, 1.0)
    investments = st.number_input('Investments',-1000000000000000.0, 100000000.0, 0.0, 1.0)
    net_from_operating_activities = st.number_input('Net from Operating Activities',min_value=0)
    net_income_loss = st.number_input('Net Income Loss',min_value=0)
    operating_lease_liability = st.number_input('Operating Lease Liability',min_value=0)
    retained_earnings = st.number_input('Retained Earnings',min_value=0)
    stockholders_equity = st.number_input('Stockholders Equity',min_value=0)
    diluted_shares_outstanding = st.number_input('Diluted Shares Outstanding',min_value=0)
    median_worker_pay = st.number_input('Median Worker Pay',min_value=0)
    ceo_salary = st.number_input('CEO Salary',min_value=0)
    revenue = st.number_input('Revenue',min_value=0)
    gross_profit = st.number_input('Gross Profit',min_value=0)
    total_debt = st.number_input('Total Debt',min_value=0)
    market_cap = st.number_input('Market Cap',min_value=0)

# Preprocess the user input
    if st.button("Get Your Prediction"):
        X = pd.DataFrame({
        'industry': [industry],
        'entity_public_float': [entity_public_float],
        'comprehensive_loss_net_of_tax': [comprehensive_loss_net_of_tax],
        'assets': [assets],
        'cash_equivalents': [cash_equivalents],
        'comprehensive_income_net_of_tax': [comprehensive_income_net_of_tax],
        'earnings_per_share_diluted': [earnings_per_share_diluted],
        'income_tax_expense_benefit': [income_tax_expense_benefit],
        'liabilities_and_stockholders_equity': [liabilities_and_stockholders_equity],
        'net_from_financing_activities': [net_from_financing_activities],
        'investments': [investments],
        'net_from_operating_activities': [net_from_operating_activities],
        'net_income_loss': [net_income_loss],
        'operating_lease_liability': [operating_lease_liability],
        'retained_earnings': [retained_earnings],
        'stockholders_equity': [stockholders_equity],
        'diluted_shares_outstanding': [diluted_shares_outstanding],
        'median_worker_pay': [median_worker_pay],
        'ceo_salary': [ceo_salary],
        'revenue': [revenue],
        'gross_profit': [gross_profit],
        'total_debt': [total_debt],
        'market_cap': [market_cap]})

        X = X
        numerical = X.select_dtypes(include = np.number)
        categorical = X.select_dtypes(include = np.object)

        cat_transformed = encoder.transform(categorical)
        col_names = ['industry_Communication Services', 'industry_Consumer Discretionary',
       'industry_Consumer Staples', 'industry_Energy', 'industry_Financials',
       'industry_Health Care', 'industry_Industrials',
       'industry_Information Technology', 'industry_Materials',
       'industry_Real Estate', 'industry_Utilities']
        
        categorical = pd.DataFrame(cat_transformed.toarray(), columns = col_names)

        # Joning dataframes
        X = pd.concat([numerical, categorical], axis=1)
        # Scaling data
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

        # Making predictions
        prediction4 = model.predict(X_scaled_df)
        return prediction4







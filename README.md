# DB Competition 2024: Seoul Apartment Price Bubble Detection and Risk Warning System

This repository contains the code for the "DB Competition 2024" project, titled "Utilizing State-Space Models and Machine Learning for Detecting Apartment Price Bubbles and Warning System in Seoul". The project aims to leverage state-space models and machine learning techniques, including logistic regression, to identify and predict apartment price bubbles in Seoul, providing valuable insights into potential market risks.

## Project Overview

This project utilizes a combination of state-space models and machine learning algorithms to analyze apartment price data in Seoul. By identifying patterns that indicate the presence of price bubbles, the system can alert stakeholders to potential risks in the housing market. The methodology includes data preprocessing, variable calculation at the city and district levels, and predictive modeling using machine learning and logistic regression.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- pandas
- numpy
- scikit-learn

pip install pandas numpy scikit-learn statsmodels
# Installation
1. Clone the repository to your local machine:
```
git clone https://github.com/yourusername/db-competition-2024.git
```
2. Navigate to the cloned repository's directory:
```
cd db-competition-2024
```
# Data Preparation
Before running the analysis, ensure your data is located in the data/ directory (you'll need to create this directory if it doesn't exist).
# Running the Code
## Data Preprocessing
Run the preprocessing script to clean and prepare the data for analysis:
```
[DB]변수 전처리.ipynb
```
## Variable Calculation
Calculate city-level and district-level variables needed for the analysis:
```
[DB]시별 변수 계산.ipynb
[DB]구별 변수 계산.ipynb
```
## Machine Learning and Logistic Regression
Execute the machine learning and logistic regression models to predict apartment price bubbles:
```
[DB]기계학습.ipynb and [DB]logistic regression.r
```

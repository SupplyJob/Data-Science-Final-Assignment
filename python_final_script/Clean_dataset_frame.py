# Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import seaborn as sns
import statsmodels.api as sm                                          # Import for regression model (Part 5)

# Tells the computer to default this as working directory 
os.chdir(r"C:\Users\billc\OneDrive\Desktop\Final Project Data Science")



# ----------------------------------------------------------------------
# Load and read excel file from Food Index dataset
food_data = pd.read_excel("Food_index.xlsx", sheet_name="Food inflation")

# Skip metadata rows and keep only useful statistics for analysis
food_df = food_data.iloc[4:].copy()

# Rename columns title
food_df.columns = ['Date', 'Food Inflation Rate (%)']
food_df = food_df[['Date', 'Food Inflation Rate (%)']]

# Remove missing values that are Nah
food_df.dropna(inplace=True)

# Convert 'Date' and 'Food Inflation' columns into proper data types
food_df['Date'] = pd.to_datetime(food_df['Date'], errors='coerce')
food_df['Food Inflation Rate (%)'] = pd.to_numeric(food_df['Food Inflation Rate (%)'], errors='coerce')
food_df.dropna(inplace=True)                                           # Remove missing values that are Nah

# Align and match dates to the start of the month
food_df['Date'] = food_df['Date'].dt.to_period('M').dt.to_timestamp()



# ----------------------------------------------------------------------
# Load and clean CPIH (Consumer Price Index) dataset
total_CPI = pd.read_csv("CPIH_Annual_rate.csv", skiprows=5)            # Skip unncessary information for data extraction                                   

# Rename columns title
total_CPI.columns = ['Date', 'Overall Consumer Price Index (%)']

# Convert 'Date' and 'Food Inflation' columns into proper data types
total_CPI['Date'] = pd.to_datetime(total_CPI['Date'], errors='coerce')

# Make sure the values from total_CPI are numeric, avoiding unit errors
total_CPI['Overall Consumer Price Index (%)'] = pd.to_numeric(total_CPI['Overall Consumer Price Index (%)'], errors='coerce')
total_CPI.dropna(inplace=True)                                         # Remove missing values that are Nah 

# Align and match dates to the start of the month
total_CPI['Date'] = total_CPI['Date'].dt.to_period('M').dt.to_timestamp()



# ----------------------------------------------------------------------
# Load and read excel file from Weekly Earnings dataset
earnings_df = pd.read_csv("Median_weekly_earnings_for_full-time_employees.csv", skiprows=12)         # Skip unncessary information for data extraction     

# Rename columns title 
earnings_df.columns = ["Year", "Full-Time", "Part-Time", "All"]

# Remove missing values that are Nah
earnings_df.dropna(inplace=True)

# Convert 'Year' into datatime type, in order to ensure consistency 
earnings_df["Year"] = pd.to_datetime(earnings_df["Year"], format="%Y")

# Check and make sure earnings values are numeric
earnings_df["Full-Time"] = pd.to_numeric(earnings_df["Full-Time"], errors = 'coerce')
earnings_df["Part-Time"] = pd.to_numeric(earnings_df["Part-Time"], errors='coerce')
earnings_df["All"] = pd.to_numeric(earnings_df["All"], errors='coerce')



# ----------------------------------------------------------------------
# Load and read excel file from Family Food Spending dataset
family_spending_df = pd.read_csv("Household_food_budget_proportion.csv")

# Check column name 
family_spending_df.columns = [
    "Year",
    "Low Income Food Spend (%)",
    "All Households Food Spend (%)"
]

# Reformat the column 'Year' by removing spaces and convert to datatime type (Using Year as unit)
family_spending_df["Year"] = family_spending_df["Year"].str.strip()
family_spending_df["Year"] = pd.to_datetime(family_spending_df["Year"].str[:4], format="%Y")            

# Ensure python to interpreted as numeric values, not string or boolean 
family_spending_df["Low Income Food Spend (%)"] = pd.to_numeric(family_spending_df["Low Income Food Spend (%)"], errors='coerce')
family_spending_df["All Households Food Spend (%)"] = pd.to_numeric(family_spending_df["All Households Food Spend (%)"], errors='coerce')
family_spending_df.dropna(inplace=True)                                  # Remove missing values that are Nah

# This new and cleaned version replaced the original dataset 
family_spending_df.to_csv("cleaned_family_food_spending.csv", index=False)



# ----------------------------------------------------------------------
# Merge food inflation and CPI for comparsion
merged_inflation = pd.merge(food_df, total_CPI, on="Date", how="inner")
print("\nMerged inflation data:\n", merged_inflation.head(10), "\n")                # Print and review the format of the dataset (10 sample)

# Merge earnings and food share data for income analysis
merged_earnings_foodshare = pd.merge(earnings_df, family_spending_df, on="Year", how="inner")
print("Merged earnings + food share data:\n", merged_earnings_foodshare.head())     # Print and review the format of the dataset



# ----------------------------------------------------------------------
# Visualization and Analysis 
# Overall Consumer Price Index(CPI) vs Food Prices (Part 1)

plt.figure(figsize=(10, 5))

# Line plot for Food Price
plt.plot(merged_inflation['Date'], merged_inflation['Food Inflation Rate (%)'], label = 'Food Inflation (%)', marker = 'x', ms = 5, color = 'aqua')

# Line plot for overall CPI
plt.plot(merged_inflation['Date'], merged_inflation['Overall Consumer Price Index (%)'], label = 'Overall CPI (%)', marker = 'o', ms = 5, color = 'limegreen')

# Titles and labels
plt.title('Comparing Food prices and CPIH (UK)', fontsize = 15, fontweight = 'bold')
plt.xlabel('Year')
plt.ylabel('Inflation Rate (%)')
plt.legend()
plt.grid(True)
plt.show()



# Supermarkets Price comparsion (Part 2)

import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from io import StringIO

# Extract data from selected URL
url = "https://www.which.co.uk/reviews/supermarkets/article/food-price-inflation-tracker-aU2oV0A46tu3"

# Bypassing websites detection as bots to avoid blocked from the webpage
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)

# Parse the HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Loop through all tables to find the inflation data
for table in soup.find_all('table'):
    table_html = str(table)
    supermarket_price_df = pd.read_html(StringIO(table_html))[0]            # [0] means get the first values and return it to the table in pandas

    # Rename columns for improving clarity
    supermarket_price_df.rename(columns={
        "Annual inflation for the three months to end November 2024": "3-Month Avg (%)",
        "Annual inflation for the month of November 2024": "Nov 2024 (%)"
    }, inplace=True)
    
    # Convert percentage strings to float
    supermarket_price_df["3-Month Avg (%)"] = supermarket_price_df["3-Month Avg (%)"].astype(str).str.replace('%', '').astype(float)
    supermarket_price_df["Nov 2024 (%)"] = supermarket_price_df["Nov 2024 (%)"].astype(str).str.replace('%', '').astype(float)

    # Additional info for the table
    supermarket_price_df["Change (%)"] = (supermarket_price_df["Nov 2024 (%)"] - supermarket_price_df["3-Month Avg (%)"]).round(1)
    supermarket_price_df["Rank"] = supermarket_price_df["Change (%)"].rank(ascending=False).astype(int)

    # Sort values for high values to low values
    supermarket_price_df.sort_values(by="Nov 2024 (%)", ascending=False, inplace=True)

    # Save to CSV
    supermarket_price_df.to_csv("supermarket_inflation.csv", index=False)
    print("Saved as 'supermarket_inflation.csv'")

    # Display table
    print("\nSupermarket Inflation Comparison:")
    print(tabulate(supermarket_price_df, headers='keys', tablefmt='fancy_grid', stralign="center", showindex=False))
    break



# Median of Weekly Earnings for employees (FULL TIME VS PART TIME) (Part 3) 

# Configure the title of x-axis to be "Year"
earnings_years =earnings_df["Year"].dt.year
earnings_full = earnings_df["Full-Time"]
earnings_part = earnings_df["Part-Time"]
earnings_all = (earnings_df["All"])

# Setting Bar width for Grouped Bar chart
bar_width = 0.3
x = np.arange(len(earnings_years))

plt.figure(figsize=(12, 6))

plt.bar(x - bar_width, earnings_full, width=bar_width , color='lightskyblue', edgecolor='black', label='Full Time')
plt.bar(x, earnings_part, width=bar_width, color='pink', edgecolor='black', label='Part Time')
plt.plot(x + bar_width, earnings_all, color='green', marker='o', linewidth=2, label='All Employees')

# Create a for loop for displaying text annotations on All Employees
for i in range (len(earnings_all)):
    plt.text(x[i] + bar_width, earnings_all.iloc[i] + 20, f"£{earnings_all.iloc[i]:.0f}", fontsize=10, color='black', ha='center')

# Titles and labels
plt.title('Median Weekly Earnings for Full-Time Employees (UK)', fontsize=15, fontweight = 'bold')
plt.xlabel('Year')
plt.ylabel('Earnings (£)')
plt.xticks(x, earnings_years, rotation = 45)
plt.legend()  
plt.grid(axis='y', linestyle='dashed', alpha=0.7)
plt.tight_layout()
plt.show()



# Share of Household Budget Spent on Food (Part 4)

# Extract data from merged family_spending_df
food_budget_years = family_spending_df["Year"]
low_income_share = family_spending_df["Low Income Food Spend (%)"]
all_households_share = family_spending_df["All Households Food Spend (%)"]

# Income gap exist? (Why low-income group pay more for food?)
income_gap = low_income_share - all_households_share 

plt.figure(figsize=(10, 6))

# Plotting food budget % over time for both household types
plt.plot(food_budget_years, low_income_share, marker='o', linestyle='-', linewidth=2,
         label='Low-Income Households', color='blueviolet')

plt.plot(food_budget_years, all_households_share, marker='s', linestyle='--', linewidth=2,
         label='All Households', color='gold')

# Fill the gap area
plt.fill_between(food_budget_years, low_income_share, all_households_share, color='lightgrey', alpha=0.5, label='Purchasing Power Gap')

# Titles and formatting
plt.title("Share of Household Budget Spent on Food", fontsize=15, fontweight = 'bold')
plt.xlabel("Year")
plt.ylabel("Percentage (%) of Budget Spent on Food")
plt.xticks(rotation=45)
plt.grid(True, linestyle='dashed', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------
# Regression modelling: Does income level correlate with how much is spent on food in the UK (Part 5)
# Load Weekly Food Basket Expenditure
food_trends_file = pd.ExcelFile("Detailed_expenditure_and_trends.xlsx")
print("Available sheets:", food_trends_file.sheet_names)

from sklearn.linear_model import LinearRegression

# Choose Sheet 3.2 from food_trends_file 
# Parse it to extract food spending by income group
food_spending_income_df = food_trends_file.parse("3.2", skiprows=30)            # Skip unncessary information for data extraction

# Clean and rename the relevant columns, make sure there is no matching error with the column title
food_spending_income_df.columns = [
    "Unused1", "Code", "Category", "Unused2", "Lowest", "Second", "Third", "Fourth",
    "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Highest", "All"
]

# Select only the columns needed for the regression modelling
food_spending_income_df = food_spending_income_df[[
    "Category", "Lowest", "Second", "Third", "Fourth", "Fifth",
    "Sixth", "Seventh", "Eighth", "Ninth", "Highest"
]]

food_spending_income_df.dropna(subset=["Category"], inplace=True)               # Skip and drop values in 'Category' that are Nah 

# Choose 6 samples of food categories to represent how normal consumers will buy in a supermarket
food_categories_regression = [
    "Milk", 
    "Chocolate", 
    "Fresh vegetables",
    "Fish and fish products",
    "Eggs",
    "Beef (fresh, chilled or frozen)"
]

# Filter selected categories
selected_income_food_df = food_spending_income_df[
    food_spending_income_df["Category"].isin(food_categories_regression)
].copy()

# Convert data to long format
long_income_food_df = selected_income_food_df.melt(
    id_vars="Category",
    var_name="Income Decile",
    value_name="Weekly Spend (£)"
)

# Mapping String variables into numeric values for modelling, in which the model only accepts numeric values  
income_decile_map = {
    "Lowest": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5,
    "Sixth": 6, "Seventh": 7, "Eighth": 8, "Ninth": 9, "Highest": 10
}
long_income_food_df["Income Decile Rank"] = long_income_food_df["Income Decile"].map(income_decile_map)

# Ensure "Weekly Spend (£)" are numeric values, avoiding errors
long_income_food_df["Weekly Spend (£)"] = pd.to_numeric(long_income_food_df["Weekly Spend (£)"], errors='coerce')
long_income_food_df.dropna(inplace=True)                                        # Skip and drop values that are Nah

# Visualize regression result 

plt.figure(figsize=(18, 10))
sns.set_style("whitegrid")

# Create a for loop to loop through all the 6 samples of food category
for i, category in enumerate(long_income_food_df["Category"].unique()):
    
    # Filter data for the current category
    category_data = long_income_food_df[long_income_food_df["Category"] == category]
    
    # Configure X and Y as independent and dependent variable respectively for regression plotting
    X = category_data[["Income Decile Rank"]]
    y = category_data["Weekly Spend (£)"]

    # Basic foundation of regression modelling 
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)                                                   # Predict values

    # Extract slope (coefficent), intercept, and R² (Extra info)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)                                               # Add the coefficient of determination, to represents variation for spending in proportion to earnings
    
    # Subplot for each category
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x="Income Decile Rank", y="Weekly Spend (£)", data=category_data, color="crimson", s=70, label="Observed")
    
    # Titles and labels
    plt.plot(X, y_pred, color="navy", linewidth=2, label="Regression Line")
    plt.title(category, fontsize=10, fontweight = 'bold')
    plt.xlabel("Income Decile Group in the UK")
    plt.ylabel("Percentage (%) of Total Expenditure")
    plt.legend()
    
      # Add regression formula and R² on each subplot 
    plt.text(
        1, max(y)*0.9,
        f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}",
        fontsize=8, color='black', bbox=dict(facecolor='lightgrey', alpha=0.7)
    )

# Main title for the whole Regression analysis
plt.suptitle("Does Income Matters in Determining Food Spending? (Regression Analysis)", fontsize=15)
plt.tight_layout()
plt.show()





## Walmart Store Sales Forecasting
![A Walmart Store](images/header.jpeg "A Walmart Store")

### Kaggle Problem Definition
Walmart is a supermarket chain in the USA. Currently, they have opened their stores all over the world.
Predicting future sales for a company like Walmart is one of the most important aspects of strategic planning. Based on the sales number they can take an informed decision for short term and long term. The future sales number also help to recruit contract employee based on sales. It will also help a Walmart store to work more efficient way.

### The Big Question - Why we need a Machine Learning approach?
They have multiple stores in various location across the world. There may have different sales pattern in different store placed in different locations. To identify this pattern and predict future sales, we should have a complete solution. Here, we can take a machine learning approach to create this complete solution. 
If we have a trained model, predicting the sales number of a store shouldn't be a tedious task to do. It will save lots of human time.

## Overview of this Blog:
- **Part-1: Kaggle Data**
- **Part-2: Exploratory Data Analysis (EDA)**
- **Part-3: Data Pre-Processing**
- **Part-4: Machine Learning Regression Models**
- **Part-5: The Model got the lowest error**
- **Part-6: Future Work**

### Part-1: Kaggle Data
We are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains many departments, and you are tasked with predicting the department-wide sales for each store.

In addition, Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. Part of the challenge presented by this competition is modelling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.

**stores.csv**<br />
This file contains anonymized information about the 45 stores, indicating the type and size of the store.

**train.csv**<br />
This is the historical training data covering 2010–02–05 to 2012–11–01. Within this file you will find the following fields:
- Store - the store number
- Dept - the department number
- Date - the week
- Weekly_Sales - sales for the given department in the given store
- IsHoliday - whether the week is a special holiday week

**test.csv**<br />
This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

**features.csv**
This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:
- Store - the store number
- Date - the week
- Temperature - the average temperature in the region
- Fuel_Price - the cost of fuel in the region
- MarkDown1–5 - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011 and is not available for all stores all the time. Any missing value is marked with an NA.
- CPI - the consumer price index
- Unemployment - the unemployment rate
- IsHoliday - whether the week is a special holiday week

There are 4 holiday mentioned in the dataset

1. Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
2. Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
3. Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
4. Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

The performance Metric of this competition is the Weighted Mean Absolute Error(**WMAE**). As per the Kaggle page, the weight of the week which has a holiday is 5 otherwise 1. That means Walmart is more focused on the holiday week and they want to have less error on that week. We also focus on that part as well and try to reduce the error.
![Evaluation metric](images/wmae.jpeg "Kaggle Evaluation metric")

### Part-2: Exploratory Data Analysis (EDA)
At first, we want to load all CSV files
```python
#Sales Data
train_data = pd.read_csv("train/train.csv")
test_data = pd.read_csv("test/test.csv")

#Features.csv
feature_data = pd.read_csv("features.csv")

#Stores.csv
store_data = pd.read_csv("stores.csv")
```
Currently, we have multiple data frames. Let join those data frames.
```python
#Join The Train data
train_all = train_data.set_index("Store").join(store_data.set_index("Store"))
train_all.reset_index(inplace=True)
train_all = train_all.set_index(["Store","Date"]).join(feature_data.set_index(["Store","Date"]))
train_all.reset_index(inplace=True)
#Train data size
print(train_all.shape)
train_all.head()
```
![Marge Train data](images/df_1_head.png "Marge Train data")

Now, check the number of NaN values for each column for both Train and Test data.


**Train Data**
```python
print(train_all.isnull().sum())
print(train_all.isnull().sum()*100.0/train_all.shape[0])
```
![Train data info](images/train_df_info.png "Train data info")
![Train data info](images/train_df_info_p.png "Train data info")

**Test Data**
```python
print(test_all.isnull().sum())
print(test_all.isnull().sum()*100.0/test_all.shape[0])
```
![Test data info](images/test_df_info.png "Test data info")
![Test data info](images/test_df_info_p.png "Test data info")

##### Observations
1. There are more than 60 per cent markdowns that are NULL. As per the competition page Markdowns are available after Nov 2011. For the data before Nov 2011, the markdowns should be zero.
2. For one-third of the test data the CPI and Unemployment. This ware depends on the location and time. But it cannot be changed drastically. So we can impute store location-wise mean of these columns.

Let's fill in the NaN values -
```python
#Replace CPI & Unemployment
test_all['CPI'] = test_all['CPI'].fillna(train_all.groupby('Store')['CPI'].transform('mean'))
test_all['Unemployment'] = test_all['Unemployment'].fillna(train_all.groupby('Store')['Unemployment'].transform('mean'))
#Replace the Markdowns with zero
train_all.fillna(0, inplace=True)
test_all.fillna(0, inplace=True)
```

#### Store Type Vs Size
```python
sns.set_style("whitegrid")
ax = sns.boxplot(x='Type', y='Size', data=train_all).set_title("Box Plot of Type vs Size")
```
![Store Type Vs Size](images/type_vs_size.png "Store Type Vs Size")

##### Observations
1. Size is very different for different types of store.

#### Highest Sales of a Week for a Department
```python
train_all_new = train_all.sort_values('Weekly_Sales',ascending=False)
train_all_new = train_all_new.head(50)
train_all_new['Week'] = train_all_new['Date'].dt.week
train_all_new['Month'] = train_all_new['Date'].dt.month
train_all_new['Year'] = train_all_new['Date'].dt.year
#Pick top 100 rows
print(tabulate(train_all_new[['Store','Dept','Size','Type','Week','Month','Year','Weekly_Sales']], headers='keys', tablefmt='psql', showindex=False))
```
![Highest Sales](images/most_sales.png "Highest Sales")

##### Observations
1. The highest sales are available on the week of Thanksgiving and the week before Christmas.
2. Most of the sales take place in the department of 7 and 72

#### Average Weekly sales over the time
```python
avg_sales = train_all.groupby('Date')[['Weekly_Sales']].mean().reset_index()
fig = plt.figure(figsize=(18,6))
plt.plot(avg_sales['Date'], avg_sales['Weekly_Sales'])
plt.xlabel("Time --->", fontsize=16)
plt.ylabel("Weekly Sales --->", fontsize=16)
plt.title("Average weekly sales over time", fontsize=18)
plt.show()
```
![Average Sales](images/average_weekly_sales.png "Average Sales")

#### Let zoom the sales number over the year
```python
avg_sales = train_all.groupby('Date')[['Weekly_Sales']].agg(['mean','median']).reset_index()
avg_sales['Week'] = avg_sales['Date'].dt.week
avg_sales_2010 = avg_sales[avg_sales['Date'].dt.year == 2010]
avg_sales_2011 = avg_sales[avg_sales['Date'].dt.year == 2011]
avg_sales_2012 = avg_sales[avg_sales['Date'].dt.year == 2012]
#Plot the model
fig = plt.figure(figsize=(18,6))
plt.plot(avg_sales_2010['Week'], avg_sales_2010['Weekly_Sales']['mean'], label='Year 2010 Mean Sales')
plt.plot(avg_sales_2010['Week'], avg_sales_2010['Weekly_Sales']['median'], label='Year 2010 Median Sales')
plt.plot(avg_sales_2011['Week'], avg_sales_2011['Weekly_Sales']['mean'], label='Year 2011 Mean Sales')
plt.plot(avg_sales_2011['Week'], avg_sales_2011['Weekly_Sales']['median'], label='Year 2011 Median Sales')
plt.plot(avg_sales_2012['Week'], avg_sales_2012['Weekly_Sales']['mean'], label='Year 2012 Mean Sales')
plt.plot(avg_sales_2012['Week'], avg_sales_2012['Weekly_Sales']['median'], label='Year 2012 Median Sales')
plt.xticks(np.arange(1, 53, step=1))
plt.grid(axis='both',color='grey', linestyle='--', linewidth=1)
plt.legend()
plt.xlabel("Week --->", fontsize=16)
plt.ylabel("Weekly Sales --->", fontsize=16)
plt.title("Average/Median weekly sales of the Different Week Over a year", fontsize=18)
```
![Average/Median Weekly Sales](images/average_weekly_sales_by_year.png "Average/Median Weekly Sales")

##### Observations
1. The Sales Number from last November to December are much bigger than a normal week.
2. The Sales number are minimum after December.
3. We can see a spike for Super Bawl.
4. In January the average sales are low.

Now, Let's try to find the pattern of Weekly sales for some departments.

```python
avg_sales_depart = train_all.groupby(['Date','Dept'])[['Weekly_Sales']].mean().reset_index()
#PLot for department 1 to 10

fig = plt.figure(figsize=(18,6))
for depart in range(1,11):
    avg_sales_depa_curr = avg_sales_depart[avg_sales_depart['Dept'] == depart]
    plt.plot(avg_sales_depa_curr['Date'], avg_sales_depa_curr['Weekly_Sales'], label='Department ' + str(depart))
    
plt.xlabel("Time --->", fontsize=16)
plt.ylabel("Weekly Sales --->", fontsize=16)
plt.title("Average weekly sales over time for department 1 to department 10",, fontsize=18)
plt.legend()
plt.show()
```
![Average Weekly Sales for dept 1 to 10](images/average_weekly_sales_by_dept_1.png "Average Weekly Sales for dept 1 to 10")

Similarly, we can plot the same data for 11 to 20 and 20 to 30

![Average Weekly Sales for dept 11 to 20](images/average_weekly_sales_by_dept_1.png "Average Weekly Sales for dept 11 to 20")
![Average Weekly Sales for dept 21 to 30](images/average_weekly_sales_by_dept_1.png "Average Weekly Sales for dept 21 to 30")

##### Observations
1. Each Department has its own behaviour.
2. For some of the department pattern of sales, the number is similar to the average sales number of all departments.
3. We should have different models for different departments.

**Here I have decided to train a model for each store and department.**

#### Average weekly sales with store category and size
In the below plot I am trying to plot average sales with store and size of the bubble is denoting the size of the store.
```python
avg_sales = train_all.groupby(['Store','Size', 'Type'])[['Weekly_Sales']].mean().reset_index()

fig = px.scatter(avg_sales, x="Store", y="Weekly_Sales",
	         size="Size", color="Type",
                 hover_name="Store", log_y=True,
                 size_max=60, title="Average Weekly Sales number with size and categoty of the store")
fig.show()
```
![Average weekly sales with store category and size](images/newplot.png "Average weekly sales with store category and size")
Here, I am trying to plot average sales with store and size of the bubble is denoting the size of the store.

##### Observations
1. Some stores are in A or B category but the sizes of those store are leas than the size of any C category store.
2. Here we can see a broad partern that the store has more size have more sale numbers.

#### Store wise Average Sales Numbers Holiday vs Non Holiday week

```python
weekly_sales_dept = train_all.groupby(["Store","IsHoliday"])[["Weekly_Sales"]].mean().reset_index()
plt.figure(figsize=(20,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Store", y="Weekly_Sales",hue="IsHoliday", data=weekly_sales_dept)
plt.title('Average Sales - per Department', fontsize=18)
plt.ylabel('Sales Number', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()
```
Here we are trying to compare average sales of a store on a holiday week and non holiday week.
![Store wise Average Sales Numbers Holiday vs Non Holiday week](images/Store_wise_Average_Sales.png "Store wise Average Sales Numbers Holiday vs Non Holiday week")

##### Observations
1. Most of the stores have better sales number in the holliday week.

#### Storewise average percentage sale
```python
weekly_sales_dept = train_all.groupby(["Store", 'Type'])[["Weekly_Sales"]].mean().reset_index()
weekly_sales_mean = weekly_sales_dept["Weekly_Sales"].sum()
weekly_sales_dept['Percentage'] = weekly_sales_dept['Weekly_Sales']/weekly_sales_mean*100
weekly_sales_dept = weekly_sales_dept.sort_values('Percentage', ascending=False).reset_index()
weekly_sales_dept['Store Number'] = str(weekly_sales_dept['Store'])

plt.figure(figsize=(24,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Store", y="Percentage", data=weekly_sales_dept, palette='deep',order=weekly_sales_dept['Store'],hue='Type', dodge=False )

for p in ax.patches:
    ax.annotate(str(format(p.get_height(), '.2f'))+"%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=12,
                   xytext = (0, -20),
                   rotation=90,
                   weight='bold',
                   color='white',
                   textcoords = 'offset points')
    
plt.title('Average Sales - per Department', fontsize=18)
plt.ylabel('Average Sales Number', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()
```
![Storewise average percentage sale](images/Storewise_average_percentage_sale.png "Storewise average percentage sale")

##### Observations
1. Some A category store has less average sales number than some B or C category.
2. Most of A category stores have more number than B or C category.

#### Department wise Average Sales Numbers Holiday vs Non Holiday week

```python
weekly_sales_dept = train_all.groupby(["Dept","IsHoliday"])[["Weekly_Sales"]].mean().reset_index()
plt.figure(figsize=(20,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Dept", y="Weekly_Sales",hue="IsHoliday", data=weekly_sales_dept)
plt.title('Average Sales - per Department', fontsize=18)
plt.grid(axis='y',color='gray', linestyle='--', linewidth=1)
plt.ylabel('Sales Number', fontsize=16)
plt.xlabel('Departments', fontsize=16)
plt.show()
```
Here we are trying to compare average sales of a department on a holiday week and non holiday week.
![Department wise Average Sales Numbers](images/average_sales_dept.png "Department wise Average Sales Numbers")

##### Observations
1. For Some departments the increase of Holiday sales are bigger than the others. 
2. But mmost of the departments the holiday sales is the range of the average sales.

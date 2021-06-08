## Walmart Store Sales Forecasting
![A Walmart Store](https://github.com/arnab000007/Walmart-Store-Sales-Forecasting/blob/main/images/header.jpeg "A Walmart Store")

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
![Evaluation metric](https://github.com/arnab000007/Walmart-Store-Sales-Forecasting/blob/main/images/wmae.jpeg "Kaggle Evaluation metric")


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/arnab000007/walmart-sales/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/arnab000007/walmart-sales/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

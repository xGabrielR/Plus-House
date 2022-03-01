# Plus House Sales

![plus](https://user-images.githubusercontent.com/75986085/155902024-c64ee122-82c2-4f5e-a9d9-f16fe93e5998.png)

<h2>0. Plus Houses Info</h2>
<p>Plus House are a real state located in south of Brazil, with around 10 employees. The company was founded by Jeorge Rodrigo.</p>
<p>What is a Real State Company ?<br>Bassicaly the company work with properties like houses in diferent styles, duples, single-family, townhouse end unit and much more.</p>
<p>In other Words, The company receives property sellers of different categories (studio, house, flat, kitnets... ), depending on the characteristics of the property, the company itself buys the property to resell on the Site. The customer search on the site or mobile application the property of your choice, perform the registration and from there it enters a process called waterfall (Visit, Imagens of choiced portfolio...).
Basically the money comes mainly for customer experience buying the property. That's why a lot of technology is important for the part of pricing these properties and even segmenting a list of possible customers who access the site.</p>

<p>During a Brainstorm meeting, a new problem raised brought up by one of the brokers, the price variation is very different from the other portfolios, causing doubt in the broker who receives these pricings and in the seller who does not understand why his property was classified in a specific price range.</p>

> *How to better price the prices of properties already enabled for sale ?*

> *How to predict the price of properties giving their characteristics ?*

<p>CEO wants to see the price of some specific properties he plans to sell in 2010.</p>

Real State Bussiness Model
Plus house has a slow business model, as a person usually buys only one property at a time and it can often be the biggest purchase of their life. That is, in the sale of a property there is a team prepared at all stages of the purchase, the marketing stage, visitation, brokers, designers, engineers and even the team that will carry out the repair of the property.</p>

**Metrics & Assumptions**<br>
- Market Share: Other enterprises in the region.
- Customers & Market Size: People over > 25 years old.
- Marketing Channel:
    1. Offline: Physical Agencies, interviews...
    2. Online: Plus Houses App, Regional Marketing.
- Customers:
    1. Main Objective is on New Customers.
- Website and App:
    1 Page Speed Score and Bouce Rate is very important! (If the site / app is bad, the customer can never come back.)

1. Customers have good experience in App or Website ?
2. Older Customers can navigate on Site or App ?
3. One customer can get back and buy another portfolio* ?
4. How is the experience of customers on buying and selling process ?
5. How is the process of visiting in the portfolios ?
6. The Img in App / Site have a good view quality ? ...

<p>The Dataset Base <a href='https://www.kaggle.com/c/house-prices-advanced-regression-techniques'>House Prices at Kaggle</a>.</p>

<p>First Deploy is Telegram Bot</p>

<!-- Link of the Telegram Gif App -->

<p>Second Deploy executable software.</p>
<p>In Dev</p>

<h2>1. Bussines Problem</h2>

<p>Plus House CEO would like to predict how much cost the properties of your choice from 2010.</p>

<h2>2. Solution Strategy & Assumptions </h2>
<h3>First CRISP Cycle</h3>

<h4>2.1. After Braimstorm Interview</h4>

> *How to better price the prices of properties already enabled for sale ?*

> *How to predict the price of properties giving their characteristics ?*

<h4>2.2. Data Product</h4>

> *A.I Model to forecast the sales at smartphone*

<ul>
  <dl>
    <dt>Data Clearing & Descriptive Statistical.</dt>
      <dd>First real step is download the dataset, import in jupyter and start in seven steps to change data types, data dimension, fillout na... At first statistic dataframe, i used simple statistic descriptions to check how my data is organized.</dd>
    <dt>Feature Engineering.</dt>
      <dd>In this step, with coggle.it to make a mind map and use the mind map to create some hypothesis list, after this list, i created some new features based on month and Lot Size.</dd>
    <dt>Data Filtering.</dt>
      <dd>Simple way to reduce dimensionality of dataset, because dataset have 81 features and aprox 1400 rows, grave problem.</dd>
    <dt>Exploration Data Analysis.</dt>
      <dd>Validation of all hypotesis list with data and individual 81 Features.</dd>
    <dt>Data Preparation.</dt>
      <dd>Prepare and Split, used base of Year 2010.</dd>
    <dt>Machine Learning Modeling.</dt>
      <dd>Selection of Four ML Models, Base, Linear and two Tree-Based.</dd>
  </dl>
</ul>

<h2>3. EDA Insight's</h2>

<p>After brainstorming and hypothesis validation, some insights appeared.</p>
<h3> Top 3 Insight's </h3>
<ul>
  <li>Plus House dont sell more per year.</li>

  ![year](https://user-images.githubusercontent.com/75986085/156092595-24c6b693-f193-49f5-b6d9-39b8f313c1b8.png)

  <li>The Excellent Quality dont have Greater prices of properties.</li>

  ![quality](https://user-images.githubusercontent.com/75986085/156092503-f3460f1c-3235-4f7d-ac2d-d3d0894fc5cc.png)
  
  <li>In Timberland Neighborhood have grater prices of properties.</li>
  
  ![timberland](https://user-images.githubusercontent.com/75986085/156092367-366bc91a-489c-4823-854a-6362057a1818.png)
  
</ul>

<h2>4. Data Preparation</h2>
<p>Used individual data preparation for feature selection.</p>
<ul>
  <dl>
    <dt>Categorical Data.</dt>
      <dd>Used the Frequency Encoding and Ordinal Encoding for all Categorical Data.</dd>
    <dt>Normalization.</dt>
      <dd>After QQplot, it was not necessary to normalize, because dont have normal distribution.</dd>
    <dt>Nature Transformation.</dt>
      <dd>Working with Sin/Cos for month data.</dd>
  </dl>
</ul>
<h3>4.1. Frequency Encoding</h3>
<p>It is an encoder method that takes into account the number of times the value appears, for example in 10 records, 5 of which are blue and red, so the frequency is .5%
</p>

<h3>4.2. QQPlot</h3>
<p>With QQPlot Quantile-Quantile Plot it is possible to observe how close the tested distribution is to a normal distribution, the normal distribution is characterized when blue line is equal to red line, there are other ways of doing this verification such as statistical tests, among others.</p>

![f](https://user-images.githubusercontent.com/75986085/156092868-891b141f-7d72-49d1-83db-6c807455f01b.png)

<h3>4.3. Feature Selection</h3>
<p>XGBoost & Random Forest Feature Importance is a fast and good way to see which feature is important, feature selection is a second way to select features for better performace of model and following the principles of Occam's Razor.</p>

![xgboost](https://user-images.githubusercontent.com/75986085/156092966-1d7074e9-187a-45f1-aa92-3e5e862a0c76.png)

1. **Overall Quall**: Suggestion of XGBoost and Random Forest and have a positive correlation.
2. **Exter Qual (Evaluates the quality of the material on the exterior)**: Suggestion of XGBoost, with Ordinal Encoder its haved a good Importance.
3. **Total Sqft**: Suggestion of XGBoost, feature engineering (living_area + bsmt).
4. **1st Floor Sqft (First Floor square feet)**: Visual Linear dependence with Sales.
5. **Total Basement Sqft**: Visual Linear dependence with Sales./
6. **Gr Sqft**: Living Area Square Feet./
7. **Year Built**: Year of Property Builted, More New, more expensive.
8. **Lot Frontage (Distance between street and property)**: Have a litle linear dependence.
9. **Garage Yr Blt**: More New Garages, more expensive the house.
10. **Condition1**: Geral condition of the house.
11. **Fireplace Qu**: Have a Litle dependencie with sales./

<h2>5. Machine Learning Models</h2>
<p>I have used three models, SVR (Support Vector Regression), Random Forest and XGBoost (Gradient boosted decision tree).</p>

![models](https://user-images.githubusercontent.com/75986085/154582560-384c54b0-c4a3-4e11-8862-5905ac12c197.png)

<p>I have selected the XGBoost than all of other two for production, in the step of hyperparameter fine tuning I used a tuning technique called Random Search and tested the trained model in the dataset with data leakage and in the dataset without data leakage. The information are in Notebook m03_machine_learningII.</p>

<p>Neural Network performace for aprox 40 epochs.</p>

![nn](https://user-images.githubusercontent.com/75986085/155723418-ae002196-8c5f-40a3-85be-0e74ba9337ea.png)


<h2>6. Bussiness Results</h2>
<p>This istep is to convert the model performace in money!!.</p><p>Below have model performace for two of the mos harder shops to forecast, there are stores where the algorithm cannot predict sales, so the RMSE error was high. MAE error be greater too, to avoid this is train more the model and work on better features. Have two columns, worst & best scenario, this columns is the sum and subtraction respectively os MAE for each model forecast.</p>

![hard_shops](https://user-images.githubusercontent.com/75986085/155026649-f00b6e31-740c-465e-b67c-ddccee4342e8.png)

<p>Below have the Sum of sales for each senario.</p>

![model_money](https://user-images.githubusercontent.com/75986085/155026940-46e5fd45-4d2c-4287-bf5e-ae2ccea0cbf8.png)

<h2>7. Model Deployment</h2>
<p>For deployment i selected Heroku for base clound 24/7h free.</p>
<p>Made a Telegram Bot and Personal '.exe' app for CFO to check the sales on smartphone and desktop.</p>

![sales](https://user-images.githubusercontent.com/75986085/155308939-12f879ae-bdde-41f7-b02d-dade281606b6.png)

![img](https://user-images.githubusercontent.com/75986085/155615627-dcbe0fd7-6116-4a91-ae17-40d4e5ee3e8b.png)

<h2>7. References</h2>
<ul>
  <li><a href='https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/'>Practical Statistics Book</li>
  <li><a href='https://www.strategyzer.com/books/business-model-generation'>Model Bussiness Book</li>
  <li><a href='https://www.docusign.com.br/blog/indicadores-do-varejo'>Retail Metrics</li>
  <li><a href='https://www.kaggle.com/bhavikapanara/frequency-encoding'>Frequency Encoding</li>
  <li><a href='https://en.wikipedia.org/wiki/Gradient_boosting'>Gradient Boosting</li>
  <li><a href='https://en.wikipedia.org/wiki/Occam%27s_razor'>Occam's Razor</li>
  <li><a href='https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/'>Random Search Tuning</li>
</ul>

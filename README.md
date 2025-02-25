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

<p>First Deploy is Web App</p>

https://user-images.githubusercontent.com/75986085/156900893-525ac63f-d614-441f-a29f-f0d586c9baa4.mp4

<a href='https://plus-house-app.herokuapp.com/'>At This Link</a>

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

<h3>Second CRISP Cycle</h3>
<p>In second Cycle, i focus on Feature Engineering creating more five Features to train the model, one of them i have droped.</p>
<p>I have used same data preparation of the First Cycle, in next Cycles i can change the encoding and create new Features.</p>

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

<p>XGBoost Feature Selection on Second Cycle.</p>

![xgb](https://user-images.githubusercontent.com/75986085/156905683-ba81374e-46f0-4152-8359-8d38cb85c1ca.png)

1. **Overall Quall**: Suggestion of XGBoost and Random Forest and have a positive correlation.
2. **Exter Qual (Evaluates the quality of the material on the exterior)**: Suggestion of XGBoost, with Ordinal Encoder its haved a good Importance.
3. **Total Sqft**: Suggestion of XGBoost, feature engineering (living_area + bsmt).
4. **Total Abv Grade**: Feature Engineering Feature.
5. **Total Bath**: Feature Engineering feature.
6. **Garage Multy Car**: Feature Engineering Feature.
7. **Land Slope**: Visual Linear dependence with Sales, people prefer houses without Slope.
8. **Bldg Type**: Type of House.
9. **Exter Cond**: Exterior of House Condition.
10. **Neighborhood**: Neighborhood of House if located.
11. **Central Air**: Have Central air or Not.
12. **Garage Finish**: Have or no a finished agarage on House.
13. **Condition2**: Geral condition of the house.
15. **Foundation**: The of Foundation of the House.
16. **Bsmt Cond**: Overall condition of Basement.
17. **Heating Qc**: Quall of method Heating (Dense Glass...)
18. **Paved Drive**: Type of Paved Driveway ( Dirt, Partial or Paved )
14. **Fireplace Qu**: Quall of Fireplace of the house.

<h2>5. Machine Learning Models</h2>
<p>I have used three models, SVR (Support Vector Regression), Random Forest and XGBoost (Gradient boosted decision tree).</p>

![models_performace](https://user-images.githubusercontent.com/75986085/156093348-95f2a54d-3fa5-4f9a-a611-a58a0a4e2c51.png)

<p>Model Performace on Second Cycle.</p>

![model_performace](https://user-images.githubusercontent.com/75986085/156905694-d0f549f1-f8a4-4c9a-b642-f8c4c81f4ecb.png)

<p>I have selected the XGBoost than all of other two for production, in the step of hyperparameter fine tuning I used a tuning technique called Random Search and tested the trained model in the dataset with data leakage and in the dataset without data leakage. The information are in Notebook m03_machine_learningII.</p>

![model_c](https://user-images.githubusercontent.com/75986085/156093370-aab37568-a172-46d8-8281-0326169e3cae.png)

<p>Second Cycle Tuned Model.</p>

![model_tunned](https://user-images.githubusercontent.com/75986085/156905716-e0627f2c-620d-4012-ad82-299967399e90.png)

<h2>6. Bussiness Results</h2>
<p>This istep is to convert the model performace in money!!.</p><p>Below have model performace for two of the mos harder shops to forecast, there are stores where the algorithm cannot predict sales, so the RMSE error was high. MAE error be greater too, to avoid this is train more the model and work on better features. Have two columns, worst & best scenario, this columns is the sum and subtraction respectively os MAE for each model forecast.</p>
<p>Below have the Sum of sales for each senario.</p>

![result](https://user-images.githubusercontent.com/75986085/156093650-12d51720-304d-4bd7-96a4-d0af7be514e3.png)

<p>In a most Realistic Scenario on Second Cycle</p>

![sales](https://user-images.githubusercontent.com/75986085/156905745-7743f164-eeb6-4ca0-8d46-d77b410268b8.png)

<p>First and Second Cycles -> Error Rate of Model</p>

![second_results](https://user-images.githubusercontent.com/75986085/156905760-00f6c735-f66c-4650-ab69-ce26f61fd57f.png)

<h2>7. Second Cycle Resume</h2>
<p>The Objective of this second Cycle is Feature Engineering and EDA focusing on Indivudual 81 individual feature analysis.</p>
<ul>
  <dl>
    <dt>Reduced MAPE error of Property 812 from 0.92 to 0.38.</dt>
    <dt>The Greater MAPE is 0.61 on Dataset, reduction of 0.33 of MAPE error rate.</dt>
  </dl>
</ul>

<h2>8. Model Deployment</h2>
<p>For deployment i selected Heroku for base clound 24/7h free.</p>
<p>Made a Streamlit App for CFO to check the sales on smartphone and desktop.</p>

<h2>9. References</h2>
<ul>
  <li><a href='https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/'>Practical Statistics Book</li>
  <li><a href='https://www.strategyzer.com/books/business-model-generation'>Model Bussiness Book</li>
  <li><a href='https://www.kaggle.com/bhavikapanara/frequency-encoding'>Frequency Encoding</li>
  <li><a href='https://en.wikipedia.org/wiki/Gradient_boosting'>Gradient Boosting</li>
  <li><a href='https://en.wikipedia.org/wiki/Occam%27s_razor'>Occam's Razor</li>
  <li><a href='https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/'>Random Search Tuning</a></li>
  <li>More at Notebook.</li>
</ul>

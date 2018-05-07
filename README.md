# kotml

kotml is aiming to be a machine learning library written purely in Kotlin. I am aiming
to provide an interface similar to sklearn's classifiers. 
The motivation behind this project was to learn Kotlin and brush up on my ML knowledge.

[TOC levels=2,2]: # " "

- [Installation](#installation)
- [Features](#features)
- [Examples](#examples)

Installation
--------
To get started, include `jcenter()` in your build gradle repositories.

Then add the following to your dependencies:
```
compile 'com.tommypacker:kotml:0.1-ALPHA'
```

You can also use kotml in your maven projects by including the following in your `pom.xml`:
```
<dependency>
  <groupId>com.tommypacker</groupId>
  <artifactId>kotml</artifactId>
  <version>0.1-ALPHA</version>
  <type>pom</type>
</dependency>
```

Features
--------
* Naive Bayes classifier
    * Gaussian
    * Multinomial
    * Bernoulli
* K Nearest Neighbors
    * Non-Weighted
    * Weighted
* Linear Regression with gradient descent (currently in alpha)
* DataContainer class to easily manipulate data given in CSV form. Splits data automatically
into testing and training data with a user defined split ratio. Also allows ignoring the first
column of datasets in case of an ID column.
* Planned support for many other popular ML models (SVM, Regression, etc).

Examples
--------
kotml currently supports three Naive Bayes classifiers(`BernoulliNB()`, `GaussianNB()`,
`MultinomialNB()`), K Nearest Neighbors (`KNN()`), and LinearRegression (`LinearRegression()`).

```kotlin
// Load Data
val spamData = Datasets.loadSpam(false, 0.8)

// Create Classifier
val BNB = BernoulliNB()

// Fit classifier to training data and training labels
BNB.fit(spamData.trainingData, spamData.trainingLabels)

// Test model and print accuracy
val accuracy = BNB.test(spamData.testData, spamData.testLabels)
println(accuracy)
```

```kotlin
// Load Regression Data
val regressionData = Datasets.loadSampleRegression()

// Create Model
val sl = LinearRegression()

// Train model for a given learning rate and number of epochs
sl.train(regressionData.trainingData, regressionData.trainingResponses, 0.0005, 10000)

// Make Predictions
val predictedValues = sl.predictValues(regressionData.trainingData)
```
More examples can be found in `examples.kt`

References
----------
APIs used in this project:
* [krangl](https://github.com/holgerbrandl/krangl): Used for dataset manipulation
* [koma](https://github.com/kyonifer/koma): Used for matrix operations

Guides to help me get started:
* https://en.wikipedia.org/wiki/Naive_Bayes_classifier
* http://scikit-learn.org/stable/modules/naive_bayes.html
* https://archive.ics.uci.edu/ml/datasets.html
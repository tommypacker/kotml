package kotml

import kotml.datasets.Datasets
import kotml.knn.KNN
import kotml.naivebayes.BernoulliNB
import kotml.naivebayes.GaussianNB
import kotml.naivebayes.MultinomialNB
import kotml.regression.LinearRegression

fun main(args: Array<String>) {
    val dataset = Datasets.loadGlass()

    // Train and test multinomial Naive Bayes classifier
    val MNB = MultinomialNB()
    MNB.fit(dataset.trainingData, dataset.trainingLabels)
    var accuracy = MNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    // Train and test Gaussian Naive Bayes classifier
    val GNB = GaussianNB()
    GNB.fit(dataset.trainingData, dataset.trainingLabels)
    accuracy = GNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    // Train and test Bernoulli Naive Bayes classifier
    val BNB = BernoulliNB()
    BNB.fit(dataset.trainingData, dataset.trainingLabels)
    accuracy = BNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    // Train and test K Nearest Neighbors classifier
    val knn = KNN(dataset.trainingData, dataset.trainingLabels, 50, false)
    accuracy = knn.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    // Train and test linear regression model
    val regressionData = Datasets.loadSampleRegression()
    val sl = LinearRegression()
    sl.train(regressionData.trainingData, regressionData.trainingResponses, 0.0005, 10000)
    val predictedValues = sl.predictValues(regressionData.trainingData)
}
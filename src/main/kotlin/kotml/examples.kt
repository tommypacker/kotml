package kotml

import kotml.datasets.Datasets
import kotml.knn.KNN
import kotml.naivebayes.BernoulliNB
import kotml.naivebayes.GaussianNB
import kotml.naivebayes.MultinomialNB

fun main(args: Array<String>) {
    val dataset = Datasets.loadGlass()

    val MNB = MultinomialNB()
    MNB.fit(dataset.trainingData, dataset.trainingLabels)
    var accuracy = MNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    val GNB = GaussianNB()
    GNB.fit(dataset.trainingData, dataset.trainingLabels)
    accuracy = GNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    val BNB = BernoulliNB()
    BNB.fit(dataset.trainingData, dataset.trainingLabels)
    accuracy = BNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)

    val knn = KNN(dataset.trainingData, dataset.trainingLabels, 50, false)
    accuracy = knn.test(dataset.testData, dataset.testLabels)
    println(accuracy)
}
package kotml

import kotml.datasets.Datasets
import kotml.knn.KNN
import kotml.naivebayes.BernoulliNB
import kotml.naivebayes.GaussianNB
import kotml.naivebayes.MultinomialNB
import kotml.regression.SimpleLinear
import kotml.utils.ContinuousDataContainer

fun main(args: Array<String>) {
    /*val dataset = Datasets.loadGlass()

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
    println(accuracy)*/

    val secondDataset = ContinuousDataContainer("src/main/resources/datasets/test.csv", false, lastColLabels = true, splitRatio = 1.0)
    val sl = SimpleLinear()
    sl.train(secondDataset.trainingData.map { it.get("x") as Double }.toDoubleArray(), secondDataset.trainingLabels.toDoubleArray(), 0.00001, 1000)
}
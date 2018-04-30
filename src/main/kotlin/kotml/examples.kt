package kotml

import kotml.naivebayes.BernoulliNB
import kotml.naivebayes.GaussianNB
import kotml.naivebayes.MultinomialNB
import kotml.utils.DataContainer

fun main(args: Array<String>) {
    val dataset = DataContainer("src/main/kotlin/kotml/datasets/spam.txt", false, 0.8)

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
}
import NaiveBayes.BernoulliNB
import NaiveBayes.GaussianNB
import NaiveBayes.MultinomialNB
import Utils.DataContainer

fun main(args: Array<String>) {
    val dataset = DataContainer("src/main/kotlin/Datasets/spam.txt", true, 0.8)

    /*val MNB = MultinomialNB()
    MNB.fit(dataset.trainingData, dataset.trainingLabels)
    var accuracy = MNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)*/

    /*val GNB = GaussianNB()
    GNB.fit(dataset.trainingData, dataset.trainingLabels)
    val accuracy = GNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)*/

    val BNB = BernoulliNB()
    BNB.fit(dataset.trainingData, dataset.trainingLabels)
    var accuracy = BNB.test(dataset.testData, dataset.testLabels)
    println(accuracy)
}
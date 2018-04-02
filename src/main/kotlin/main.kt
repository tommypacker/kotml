import NaiveBayes.GaussianNB
import NaiveBayes.MultinomialNB
import Utils.DataRow
import Utils.DataTransformer
import krangl.DataFrame
import krangl.readCSV

fun main(args: Array<String>) {
    // Read Data from CSV
    /*val dataset = DataFrame.readCSV("src/main/kotlin/breast-cancer.txt")

    // Format Data
    val data = mutableListOf<DataRow>()
    val labels = mutableListOf<String>()
    DataTransformer.transformDataframe(dataset, data, labels, false)

    // Split Data
    val splits = DataTransformer.splitDataset(data, labels, .7)
    val trainingData = splits.first.first
    val trainingLabels = splits.first.second
    val testData = splits.second.first
    val testLabels = splits.second.second

    // Train Model
    val NB = GaussianNB()
    NB.fit(trainingData, trainingLabels)
    val predictions = NB.getPredictions(trainingData)
    println(predictions)

    // Test Model Accuracy
    val accuracy = NB.test(testData, testLabels)
    print(accuracy)*/

    val dataset = DataFrame.readCSV("src/main/kotlin/test.txt")

    // Format Data
    val data = mutableListOf<DataRow>()
    val labels = mutableListOf<String>()
    DataTransformer.transformDataframe(dataset, data, labels, true)

    // Split Data
    val splits = DataTransformer.splitDataset(data, labels, .7)
    val trainingData = splits.first.first
    val trainingLabels = splits.first.second
    val testData = splits.second.first
    val testLabels = splits.second.second

    // Train Model
    val NB = MultinomialNB()
    NB.fit(trainingData, trainingLabels)
    val predictions = NB.getPredictions(testData)
    println(testData)
    println(predictions)
}
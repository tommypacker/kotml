import Utils.DataTransformer
import krangl.DataFrame
import krangl.readCSV

fun main(args: Array<String>) {
    // Read Data from CSV
    val dataset = DataFrame.readCSV("src/main/kotlin/breast-cancer.txt")

    // Format Data
    val data = mutableListOf<Map<String, Any?>>()
    val labels = mutableListOf<String>()
    DataTransformer.transformDataframe(dataset, data, labels, true)

    // Split Data
    val splits = DataTransformer.splitDataset(data, labels, .70)
    val trainingData = splits.first.first
    val trainingLabels = splits.first.second
    val testData = splits.second.first
    val testLabels = splits.second.second

    // Train Model
    val NB = GaussianNB(trainingData, trainingLabels)
    NB.fit()

    // Test Model Accuracy
    val accuracy = NB.test(testData, testLabels)
    print(accuracy)
}
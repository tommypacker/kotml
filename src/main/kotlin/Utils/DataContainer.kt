package Utils

import krangl.DataFrame
import krangl.readCSV

class DataContainer (filePath: String, ignoreFirstCol: Boolean, splitRatio: Double) {

    val data: MutableList<DataRow>
    val labels: MutableList<String>
    val trainingData: MutableList<DataRow>
    val trainingLabels: MutableList<String>
    val testData: MutableList<DataRow>
    val testLabels: MutableList<String>

    init {
        val dataset = DataFrame.readCSV(filePath)
        this.data = mutableListOf()
        this.labels = mutableListOf()
        DataTransformer.transformDataframe(dataset, data, labels, ignoreFirstCol)

        val splits = DataTransformer.splitDataset(data, labels, splitRatio)
        this.trainingData = splits.first.first
        this.trainingLabels = splits.first.second
        this.testData = splits.second.first
        this.testLabels = splits.second.second
    }
}
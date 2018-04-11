package Utils

import krangl.DataFrame
import krangl.readCSV

class DataContainer (filePath: String, ignoreFirstCol: Boolean, splitRatio: Double) {

    val data: Array<DataRow>
    val labels: Array<String>
    val trainingData: Array<DataRow>
    val trainingLabels: Array<String>
    val testData: Array<DataRow>
    val testLabels: Array<String>

    init {
        val dataset = DataFrame.readCSV(filePath)
        val tempData = mutableListOf<DataRow>()
        val tempLabels = mutableListOf<String>()
        DataTransformer.transformDataframe(dataset, tempData, tempLabels, ignoreFirstCol)
        this.data = tempData.toTypedArray()
        this.labels = tempLabels.toTypedArray()

        val splits = DataTransformer.splitDataset(data, labels, splitRatio)
        this.trainingData = splits.first.first
        this.trainingLabels = splits.first.second
        this.testData = splits.second.first
        this.testLabels = splits.second.second
    }
}
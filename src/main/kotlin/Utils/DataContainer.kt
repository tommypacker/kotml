package Utils

import krangl.DataFrame
import krangl.readCSV

/**
 *  This class is used as an abstraction layer to cleanly hold training and testing data.
 *  It uses krangl to read in csv's from a file, then converts that into an array of Datarows.
 *  Alternatively, users can pass in an array of Datarows and strings (labels) to use as the data.
 */
class DataContainer (filePath: String?, ignoreFirstCol: Boolean, splitRatio: Double,
                     data: Array<DataRow> = emptyArray(), labels: Array<String> = emptyArray()) {

    val data: Array<DataRow>
    val labels: Array<String>
    val trainingData: Array<DataRow>
    val trainingLabels: Array<String>
    val testData: Array<DataRow>
    val testLabels: Array<String>

    init {
        if (filePath != null) {
            val dataset = DataFrame.readCSV(filePath)
            val tempData = mutableListOf<DataRow>()
            val tempLabels = mutableListOf<String>()
            DataTransformer.transformDataframe(dataset, tempData, tempLabels, ignoreFirstCol)
            this.data = tempData.toTypedArray()
            this.labels = tempLabels.toTypedArray()
        } else {
            this.data = data
            this.labels = labels
        }
        val splits = DataTransformer.splitDataset(this.data, this.labels, splitRatio)
        this.trainingData = splits.first.first
        this.trainingLabels = splits.first.second
        this.testData = splits.second.first
        this.testLabels = splits.second.second
    }
}
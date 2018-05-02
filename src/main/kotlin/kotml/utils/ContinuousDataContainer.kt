package kotml.utils

import krangl.DataFrame
import krangl.readCSV


/**
 *  This class is used as an abstraction layer to cleanly hold training and testing data.
 *  It uses krangl to read in csv's from a file, then converts that into an array of Datarows.
 *  Alternatively, users can pass in an array of Datarows and doubles (labels) to use as the data.
 */
class ContinuousDataContainer (filePath: String?, ignoreFirstCol: Boolean, splitRatio: Double = 0.7,
                             data: Array<DataRow> = emptyArray(), labels: Array<Double> = emptyArray(),
                             lastColLabels: Boolean = false) {

    val data: Array<DataRow>
    val labels: Array<Double>
    val trainingData: Array<DataRow>
    val trainingLabels: Array<Double>
    val testData: Array<DataRow>
    val testLabels: Array<Double>

    init {
        if (filePath != null) {
            val dataset = DataFrame.readCSV(filePath)
            val tempData = mutableListOf<DataRow>()
            val tempLabels = mutableListOf<Any>()
            DataTransformer.transformDataframe(dataset, tempData, tempLabels, ignoreFirstCol)
            this.data = tempData.toTypedArray()
            if (lastColLabels) {
                this.labels = DataTransformer.extractLastColumn(dataset).values().map { it as Double }.toTypedArray()
            } else {
                this.labels = tempLabels.toTypedArray().map { it as Double }.toTypedArray()
            }
        } else {
            this.data = data
            this.labels = labels
        }
        val splits = DataTransformer.splitDataset(this.data, this.labels, splitRatio)
        this.trainingData = splits.first.first
        this.trainingLabels = splits.first.second.map { it as Double }.toTypedArray()
        this.testData = splits.second.first
        this.testLabels = splits.second.second.map { it as Double }.toTypedArray()
    }
}
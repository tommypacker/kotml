package kotml.utils

import koma.create
import koma.eye
import krangl.DataFrame
import krangl.readCSV


/**
 *  This class is used as an abstraction layer to cleanly hold training and testing data for regression problems.
 *  It uses krangl to read in csv's from a file, then converts that into an array of DataRows.
 *  Alternatively, users can pass in an array of Datarows and doubles (labels) to use as the data.
 */
class ContinuousDataset (filePath: String?, ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7,
                         data: Array<DataRow> = emptyArray(), labels: Array<Double> = emptyArray(),
                         lastColLabels: Boolean = false) {

    val data: Array<DataRow>
    val labels: Array<Double>
    var trainingData = eye(1)
    var testData = eye(1)
    val trainingResponses: DoubleArray
    val testResponses: DoubleArray

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
        
        // Convert input variables to a Matrix<Double>
        this.trainingData = create(splits.first.first.map { it.values.map { it as Double }.toDoubleArray() }.toTypedArray())
        this.testData = create(splits.second.first.map { it.values.map { it as Double }.toDoubleArray() }.toTypedArray())
        
        // Convert explanatory vars to a DoubleArray
        this.trainingResponses = splits.first.second.map { it as Double }.toDoubleArray()
        this.testResponses = splits.second.second.map { it as Double }.toDoubleArray()
    }
}
package Utils

import krangl.DataFrame
import java.util.*

class DataTransformer {
    companion object {

        /**
         * Converts a DataFrame into a list of data rows (for features) and strings (labels)
         */
        fun transformDataframe(rawData: DataFrame, data: MutableList<Map<String, Any?>>, labels: MutableList<String>, ignoreFirstCol: Boolean) {
            val numCols = rawData.ncol
            val firstColName = rawData.cols.get(0).name
            val labelName = rawData.cols.get(numCols-1).name

            for (i in 0..rawData.nrow-1) {
                val curRow = rawData.row(i)
                labels.add(curRow.get(labelName).toString())
                if (ignoreFirstCol) {
                    data.add(curRow.minus(labelName).minus(firstColName))
                } else {
                    data.add(curRow.minus(labelName))
                }

            }
        }

        /**
         * Split given data into training vs testing
         */
        fun splitDataset(data: MutableList<Map<String, Any?>>, labels: MutableList<String>, splitRatio: Double) :
                Pair<Pair<MutableList<Map<String, Any?>>, MutableList<String>>, Pair<MutableList<Map<String, Any?>>, MutableList<String>>> {
            val trainSize = (data.size * splitRatio).toInt()
            val trainingData = mutableListOf<Map<String, Any?>>()
            val trainDataLabels = mutableListOf<String>()

            // Copy Data
            val copyData = mutableListOf<Map<String, Any?>>()
            val copyLabels = mutableListOf<String>()
            for (i in 0..data.size-1) {
                copyData.add(data.get(i))
                copyLabels.add(labels.get(i))
            }

            // Add to training data
            while (trainingData.size < trainSize) {
                val index = (0..copyData.size).random()
                trainingData.add(copyData.get(index))
                trainDataLabels.add(copyLabels.get(index))
                copyData.removeAt(index)
                copyLabels.removeAt(index)
            }

            // Test data is the leftovers
            val testData = copyData
            val testLabels = copyLabels

            val trainPair = Pair(trainingData, trainDataLabels)
            val testPair = Pair(testData, testLabels)
            return Pair(trainPair, testPair)
        }

        fun ClosedRange<Int>.random() =
                Random().nextInt(endInclusive - start) +  start
    }
}
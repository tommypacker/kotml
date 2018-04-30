package kotml.utils

import krangl.DataFrame
import java.util.*

typealias DataRow = Map<String, Any?>
typealias DataSplits = Pair<Pair<Array<DataRow>, Array<String>>, Pair<Array<DataRow>, Array<String>>>
typealias Summary = Pair<Double, Double>

class DataTransformer {
    companion object {
        /**
         *  Converts a DataFrame into a list of data rows (for features) and strings (labels)
         *  All feature values are converted to doubles, and rows with missing data are dropped
         */
        fun transformDataframe(rawData: DataFrame, data: MutableList<DataRow>, labels: MutableList<String>, ignoreFirstCol: Boolean) {
            val numCols = rawData.ncol
            val firstColName = rawData.cols.get(0).name
            val labelName = rawData.cols.get(numCols-1).name

            for (i in 0..rawData.nrow-1) {
                var curRow = rawData.row(i)
                if (ignoreFirstCol) {
                    curRow = curRow.minus(firstColName)
                }
                var shouldSkipRow = false
                for (feature in curRow.minus(labelName).keys) {
                    var value = curRow.get(feature)
                    if (value is String) {
                        if (value.toDoubleOrNull() == null) {
                            shouldSkipRow = true
                            break
                        } else {
                            value = value.toDouble()
                        }
                    } else if (value is Int) {
                        value = value.toDouble()
                    }
                    curRow = curRow.minus(feature).plus(Pair(feature, value))
                }
                if (shouldSkipRow) continue
                labels.add(curRow.get(labelName).toString())
                data.add(curRow.minus(labelName))
            }
        }

        /**
         *  Split given data into training vs testing based on given split ration
         *  Will randomly choose which rows to add to the testing set
         */
        fun splitDataset(data: Array<DataRow>, labels: Array<String>, splitRatio: Double) : DataSplits {
            val trainSize = (data.size * splitRatio).toInt()
            val trainingData = mutableListOf<DataRow>()
            val trainDataLabels = mutableListOf<String>()

            // Copy Data
            val testData = data.toMutableList()
            val testLabels = labels.toMutableList()

            // Add to training data
            while (trainingData.size < trainSize) {
                val index = (0..testData.size).random()
                trainingData.add(testData.removeAt(index))
                trainDataLabels.add(testLabels.removeAt(index))
            }

            // Package up data and return
            val trainPair = Pair(trainingData.toTypedArray(), trainDataLabels.toTypedArray())
            val testPair = Pair(testData.toTypedArray(), testLabels.toTypedArray())
            return Pair(trainPair, testPair)
        }

        /**
         *  Extension function that returns a random integer within a range
         *  Inspired by https://stackoverflow.com/questions/45685026/how-can-i-get-a-random-number-in-kotlin
         */
        fun ClosedRange<Int>.random() =
                Random().nextInt(endInclusive - start) +  start
    }
}
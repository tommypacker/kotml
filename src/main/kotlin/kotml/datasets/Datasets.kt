package kotml.datasets

import kotml.utils.MatrixDataHolder
import kotml.utils.ArrayDataHolder

private const val DATASETSPATH = "src/main/resources/datasets/"

class Datasets {
    companion object {
        fun loadIris(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ArrayDataHolder {
            return ArrayDataHolder(DATASETSPATH + "iris.csv", ignoreFirstCol, splitRatio)
        }

        fun loadSpam(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ArrayDataHolder {
            return ArrayDataHolder(DATASETSPATH + "spam.txt", ignoreFirstCol, splitRatio)
        }

        fun loadBreastCancer(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ArrayDataHolder {
            return ArrayDataHolder(DATASETSPATH + "breast-cancer.txt", ignoreFirstCol, splitRatio)
        }

        fun loadGlass(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ArrayDataHolder {
            return ArrayDataHolder(DATASETSPATH + "glass.txt", ignoreFirstCol, splitRatio)
        }

        fun loadSampleRegression(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : MatrixDataHolder {
            return MatrixDataHolder(DATASETSPATH + "regression.csv", lastColLabels = true, splitRatio = 0.9)
        }
    }
}
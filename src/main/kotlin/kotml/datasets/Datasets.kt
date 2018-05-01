package kotml.datasets

import kotml.utils.DataContainer

private const val DATASETSPATH = "src/main/resources/datasets/"

class Datasets {
    companion object {
        fun loadIris(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DataContainer {
            return DataContainer(DATASETSPATH + "iris.csv", ignoreFirstCol, splitRatio)
        }

        fun loadSpam(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DataContainer {
            return DataContainer(DATASETSPATH + "spam.txt", ignoreFirstCol, splitRatio)
        }

        fun loadBreastCancer(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DataContainer {
            return DataContainer(DATASETSPATH + "breast-cancer.txt", ignoreFirstCol, splitRatio)
        }

        fun loadGlass(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DataContainer {
            return DataContainer(DATASETSPATH + "glass.txt", ignoreFirstCol, splitRatio)
        }
    }
}
package kotml.datasets

import kotml.utils.ContinuousDataContainer
import kotml.utils.DiscreteDataContainer

private const val DATASETSPATH = "src/main/resources/datasets/"

class Datasets {
    companion object {
        fun loadIris(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataContainer {
            return DiscreteDataContainer(DATASETSPATH + "iris.csv", ignoreFirstCol, splitRatio)
        }

        fun loadSpam(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataContainer {
            return DiscreteDataContainer(DATASETSPATH + "spam.txt", ignoreFirstCol, splitRatio)
        }

        fun loadBreastCancer(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataContainer {
            return DiscreteDataContainer(DATASETSPATH + "breast-cancer.txt", ignoreFirstCol, splitRatio)
        }

        fun loadGlass(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataContainer {
            return DiscreteDataContainer(DATASETSPATH + "glass.txt", ignoreFirstCol, splitRatio)
        }

        fun loadRegression(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ContinuousDataContainer {
            return ContinuousDataContainer(DATASETSPATH + "regression.csv", lastColLabels = true, splitRatio = 0.9)
        }
    }
}
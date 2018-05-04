package kotml.datasets

import kotml.utils.ContinuousDataset
import kotml.utils.DiscreteDataset

private const val DATASETSPATH = "src/main/resources/datasets/"

class Datasets {
    companion object {
        fun loadIris(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataset {
            return DiscreteDataset(DATASETSPATH + "iris.csv", ignoreFirstCol, splitRatio)
        }

        fun loadSpam(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataset {
            return DiscreteDataset(DATASETSPATH + "spam.txt", ignoreFirstCol, splitRatio)
        }

        fun loadBreastCancer(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataset {
            return DiscreteDataset(DATASETSPATH + "breast-cancer.txt", ignoreFirstCol, splitRatio)
        }

        fun loadGlass(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : DiscreteDataset {
            return DiscreteDataset(DATASETSPATH + "glass.txt", ignoreFirstCol, splitRatio)
        }

        fun loadSampleRegression(ignoreFirstCol: Boolean = false, splitRatio: Double = 0.7) : ContinuousDataset {
            return ContinuousDataset(DATASETSPATH + "regression.csv", lastColLabels = true, splitRatio = 0.9)
        }
    }
}
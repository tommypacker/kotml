import Utils.DataTransformer
import Utils.MathHelper
import krangl.DataFrame
import krangl.DataFrameRow
import java.util.Random

class GaussianNB(rawData: DataFrame) {

    var data: MutableList<DataFrameRow>
    var labels: MutableList<String>
    var trainingData: MutableList<DataFrameRow>
    var trainDataLabels: MutableList<String>
    var testData: MutableList<DataFrameRow>
    var testLabels: MutableList<String>

    init {
        this.data = mutableListOf()
        this.labels = mutableListOf()
        this.trainingData = mutableListOf()
        this.trainDataLabels = mutableListOf()
        this.testData = mutableListOf()
        this.testLabels = mutableListOf()
        DataTransformer.transformDataframe(rawData, this.data, this.labels)
    }

    /**
     * Fit our model to the dataset
     */
    fun fit() {
        splitDataset(0.70)
        val summaries = summarizeByClass()
        val predictions = getPredictions(summaries, testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        println(accuracy)
    }

    /**
     * Split given data into training vs testing
     */
    fun splitDataset(splitRatio: Double) {
        val trainSize = (this.data.size * splitRatio).toInt()

        // Copy Data
        val copyData = mutableListOf<DataFrameRow>()
        val copyLabels = mutableListOf<String>()
        for (i in 0..this.data.size-1) {
            copyData.add(this.data.get(i))
            copyLabels.add(this.labels.get(i))
        }

        // Add to training data
        while (trainingData.size < trainSize) {
            val index = (0..copyData.size).random()
            trainingData.add(copyData.get(index))
            trainDataLabels.add(copyLabels.get(index))
            copyData.removeAt(index)
            copyLabels.removeAt(index)
        }

        this.testData = copyData
        this.testLabels = copyLabels
    }

    fun separateByLabels() : HashMap<String, MutableList<DataFrameRow>> {
        val res = HashMap<String, MutableList<DataFrameRow>>()
        for (i in 0..this.trainingData.size-1) {
            val row = this.trainingData.get(i)
            val labelVal = this.trainDataLabels.get(i)
            if (!res.containsKey(labelVal)) {
                res.put(labelVal, mutableListOf())
            }
            res.get(labelVal)?.add(row)
        }
        return res
    }

    fun separateFeatures(dataCols: MutableList<DataFrameRow>) : HashMap<String, DoubleArray> {
        val featureMap = HashMap<String, DoubleArray>()
        val cols = this.trainingData.get(0).keys
        for (colName in cols) {
            val colVals = DoubleArray(dataCols.size)

            // Need ability to slice by index rather than manually doing this
            for (i in 0..dataCols.size-1) {
                colVals[i] = dataCols.get(i).get(colName) as Double
            }

            featureMap.put(colName, colVals)
        }
        return featureMap
    }

    fun summarizeFeatures(features: HashMap<String, DoubleArray>) : HashMap<String, Pair<Double, Double>> {
        val summaries = HashMap<String, Pair<Double, Double>>()
        for (featureName in features.keys) {
            val featureVals = features.get(featureName)
            if (featureVals != null) {
                summaries.put(featureName, Pair(featureVals.average(), MathHelper.stdev(featureVals)))
            }
        }
        return summaries
    }

    fun summarizeByClass() : HashMap<String, HashMap<String, Pair<Double, Double>>> {
        val res = HashMap<String, HashMap<String, Pair<Double, Double>>>()
        val labelSepData = separateByLabels()

        // Iterate through each label value
        for (labelVal in labelSepData.keys) {
            val labelSummary = HashMap<String, Pair<Double, Double>>()
            val labelValRows = labelSepData.get(labelVal)

            // Mapping of feature to values
            val featureMap = separateFeatures(labelValRows!!)
            val summaryMap = summarizeFeatures(featureMap)
            for (feature in summaryMap.keys) {
                //println("Summary for " + labelVal + "." + feature + ": " + summaryMap.get(feature))
                labelSummary.put(feature, summaryMap.get(feature)!!)
            }
            res.put(labelVal, labelSummary)
        }

        return res
    }

    fun calculateClassProbabilities(summaries: HashMap<String, HashMap<String, Pair<Double, Double>>>,
                                    inputVector: DataFrameRow) : HashMap<String, Double> {
        val res = HashMap<String, Double>()
        for (labelVal in summaries.keys) {
            var classProbability = 1.0
            val labelFeatureSummary = summaries.get(labelVal)
            for (featureName in labelFeatureSummary!!.keys) {
                val summary = labelFeatureSummary.get(featureName)
                val mean = summary!!.first
                val stdev = summary.second
                val x = inputVector.get(featureName) as Double
                classProbability *= MathHelper.calculateProbability(x, mean, stdev)
            }
            res.put(labelVal, classProbability)
        }
        return res
    }

    fun predict(summaries: HashMap<String, HashMap<String, Pair<Double, Double>>>,
                inputVector: DataFrameRow) : String {
        val probabilties = calculateClassProbabilities(summaries, inputVector)
        var bestLabel = ""
        var bestProb = -1.0
        for (labelName in probabilties.keys) {
            val curProb = probabilties.get(labelName)
            if (curProb!! > bestProb) {
                bestProb = curProb
                bestLabel = labelName
            }
        }
        return bestLabel
    }

    fun getPredictions(summaries: HashMap<String, HashMap<String, Pair<Double, Double>>>,
                       testData: MutableList<DataFrameRow>) : MutableList<String> {
        var res = mutableListOf<String>()
        val numRows = testData.size
        for (i in 0..numRows-1) {
            val prediction = predict(summaries, testData.get(i))
            res.add(prediction)
        }
        return res
    }

    fun ClosedRange<Int>.random() =
            Random().nextInt(endInclusive - start) +  start
}
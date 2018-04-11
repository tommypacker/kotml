package NaiveBayes

import Utils.DataRow
import Utils.MathHelper
import Utils.Summary

class GaussianNB() {

    var data: Array<DataRow>
    var labels: Array<String>
    var model: HashMap<String, HashMap<String, Summary>>
    var priors: HashMap<String, Double>
    var totalRows: Int = 0

    init {
        this.data = arrayOf()
        this.labels = arrayOf()
        this.priors = HashMap()
        this.model = HashMap()
    }

    /**
     * Fit model to given data and labels
     */
    fun fit(data: Array<DataRow>, labels: Array<String>) {
        this.data = data
        this.labels = labels
        this.totalRows = data.size
        this.model = trainModel()
    }

    /**
     * Test our model by making predictions and comparing them to the actual labels
     */
    fun test(testData: Array<DataRow>, testLabels: Array<String>) : Double {
        val predictions = getPredictions(testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        return accuracy
    }

    /**
     * Make predictions on the test data based on our model
     */
    fun getPredictions(testData: Array<DataRow>) : Array<String> {
        val res = mutableListOf<String>()
        val numRows = testData.size
        for (i in 0..numRows-1) {
            val prediction = predict(testData.get(i))
            res.add(prediction)
        }
        return res.toTypedArray()
    }

    /**
     * Train our Gaussian NB model by calculating summaries for each class in our dataset
     * Each summary consists of the mean and stdev of a feature for a given class
     * More info on the model can be found here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
     */
    private fun trainModel() : HashMap<String, HashMap<String, Summary>> {
        val res = HashMap<String, HashMap<String, Summary>>()
        val classSeparatedData = separateByClass()

        // Calculate Priors
        for (classVal in classSeparatedData.keys) {
            val classCount = classSeparatedData.get(classVal)!!.size
            this.priors.put(classVal, classCount.toDouble() / this.totalRows)
        }

        // Iterate through each label value
        for (classVal in classSeparatedData.keys) {
            val classSummary = HashMap<String, Summary>()
            val dataForClass = classSeparatedData.get(classVal)

            // Mapping of feature to values
            val featureMap = separateFeatures(dataForClass!!)
            val summaryMap = summarizeFeatures(featureMap)
            for (feature in summaryMap.keys) {
                //println("Summary for " + labelVal + "." + feature + ": " + summaryMap.get(feature))
                classSummary.put(feature, summaryMap.get(feature)!!)
            }
            res.put(classVal, classSummary)
        }

        return res
    }

    /**
     * Separates data into a mapping of class value to datarows belonging to that class
     */
    private fun separateByClass() : HashMap<String, MutableList<DataRow>> {
        val res = HashMap<String, MutableList<DataRow>>()
        for (i in 0..this.data.size-1) {
            val row = this.data.get(i)
            val classVal = this.labels.get(i)
            if (!res.containsKey(classVal)) {
                res.put(classVal, mutableListOf())
            }
            res.get(classVal)?.add(row)
        }
        return res
    }

    /**
     * Returns a list of each feature column mapped to a list of its values
     */
    private fun separateFeatures(dataCols: MutableList<DataRow>) : HashMap<String, DoubleArray> {
        val featureMap = HashMap<String, DoubleArray>()
        val cols = this.data.get(0).keys
        for (colName in cols) {
            val colVals = DoubleArray(dataCols.size)

            // Need ability to slice by index rather than manually doing this
            for (i in 0..dataCols.size-1) {
                val curRow = dataCols.get(i).get(colName) as Double
                colVals[i] = curRow
            }

            featureMap.put(colName, colVals)
        }
        return featureMap
    }

    /**
     * Calculate the mean and stdev of list of feature values
     */
    private fun summarizeFeatures(features: HashMap<String, DoubleArray>) : HashMap<String, Summary> {
        val summaries = HashMap<String, Summary>()
        for (featureName in features.keys) {
            val featureVals = features.get(featureName)
            if (featureVals != null) {
                summaries.put(featureName, Pair(featureVals.average(), MathHelper.stdev(featureVals)))
            }
        }
        return summaries
    }

    /**
     * Calculate P(x|classVal) based on Gaussian Distribution
     * where x is the feature value and classVal is the given class
     */
    private fun calculateClassProbabilities(summaries: HashMap<String, HashMap<String, Summary>>,
                                    inputVector: DataRow) : HashMap<String, Double> {
        val res = HashMap<String, Double>()
        for (classVal in summaries.keys) {
            var classProbability = 0.0
            val featureSummary = summaries.get(classVal)
            for (featureName in featureSummary!!.keys) {
                val summary = featureSummary.get(featureName)
                val mean = summary!!.first
                val stdev = summary.second
                val x = inputVector.get(featureName) as Double
                classProbability += Math.log(MathHelper.calculateGaussian(x, mean, stdev))
            }
            // Add class prior
            classProbability += Math.log(this.priors.get(classVal)!!)
            res.put(classVal, classProbability)
        }
        return res
    }

    /**
     * Predict by taking maximum a posteriori estimate (MAP)
     */
    private fun predict(inputVector: DataRow) : String {
        val probabilties = calculateClassProbabilities(this.model, inputVector)
        var bestLabel = ""
        var bestProb = Int.MIN_VALUE.toDouble()
        for (className in probabilties.keys) {
            val curProb = probabilties.get(className)
            if (curProb!! > bestProb) {
                bestProb = curProb
                bestLabel = className
            }
        }
        return bestLabel
    }
}
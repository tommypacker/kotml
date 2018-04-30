package kotml.naivebayes

import kotml.utils.DataRow
import kotml.utils.MathHelper

class MultinomialNB (val alpha: Double = 1.0){
    var data = arrayOf<DataRow>()
    var labels = arrayOf<String>()
    var labelSet = setOf<String>()
    var priors = hashMapOf<String, Double>()
    var model = mapOf<String, Map<String, Double>>()
    var numRows = 0

    /**
     *  Fit model to given data and labels
     */
    fun fit(data: Array<DataRow>, labels: Array<String>) {
        this.data = data
        this.labels = labels
        this.labelSet = labels.toHashSet()
        this.numRows = data.size
        this.model = trainModel()
    }

    /**
     *  Test our model by making predictions and comparing them to the actual labels
     */
    fun test(testData: Array<DataRow>, testLabels: Array<String>) : Double {
        val predictions = getPredictions(testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        return accuracy
    }

    /**
     *  Make predictions on the test data based on our model
     */
    fun getPredictions(testData: Array<DataRow>) : Array<String> {
        val res = mutableListOf<String>()
        for (row in testData) {
            res.add(predict(row))
        }
        return res.toTypedArray()
    }

    /**
     *  Trains our model using the Multinomial naive bayes formula
     *  We need to calculate the probability of a feature in a class based on number of occurences
     *  For more info: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
     */
    private fun trainModel() : Map<String, Map<String, Double>> {
        val res = HashMap<String, Map<String, Double>>()
        val sepClassData = separateByClass()

        // n = number of feature columns
        val n = this.data[0].keys.size

        for (classVal in sepClassData.keys) {
            // Get all docs for a class
            val classData = sepClassData.get(classVal)

            // Calculate Priors
            this.priors.put(classVal, classData!!.size.toDouble()/this.numRows)

            // Aggregate feature counts per class and total number of feature counts
            val aggregatedData = aggregateCountsPerClass(classData)
            val classProbabilties = HashMap<String, Double>()
            var totalNumfeatures = 0.0
            for (featureName in aggregatedData.keys) {
                totalNumfeatures += aggregatedData.get(featureName)!!
            }

            // Calculate feature probabilities conditioned on given class: p(w_i|class)
            for (featureName in aggregatedData.keys) {
                // Use LaPlace smoothing when calculating theta values
                classProbabilties.put(featureName, (aggregatedData.get(featureName)!!.toDouble() + alpha) / (totalNumfeatures + (alpha * n)))
            }
            res.put(classVal, classProbabilties)
        }
        return res
    }

    /**
     *  Make predictions based on the multinomial model and take the MAP estimate.
     *  The model can be found here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
     */
    private fun predict(document: DataRow) : String {
        var bestProb = Int.MIN_VALUE.toDouble()
        var bestLabel = ""

        for (classVal in this.labelSet) {
            var likelihood = 0.0
            for (feature in document.keys) {
                val featureCount = document.get(feature) as Double
                val conditionalProb = this.model.get(classVal)!!.get(feature)!!
                likelihood += Math.log(conditionalProb) * featureCount
            }
            likelihood += Math.log(this.priors.get(classVal)!!)
            if (likelihood > bestProb) {
                bestProb = likelihood
                bestLabel = classVal
            }
        }
        return bestLabel
    }

    /**
     *  Counts total number of occurrences of a feature in a given class
     */
    private fun aggregateCountsPerClass(classData: MutableList<DataRow>) : Map<String, Double> {
        val res = HashMap<String, Double>()
        for (row in classData) {
            for(featureName in row.keys) {
                val count = row.get(featureName) as Double
                res.put(featureName, res.getOrDefault(featureName, 0.0) + count)
            }
        }
        return res
    }

    /**
     *  Separates data into a mapping of class value to datarows belonging to that class
     */
    private fun separateByClass() : Map<String, MutableList<DataRow>> {
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
}
package kotml.NaiveBayes

import kotml.Utils.DataRow
import kotml.Utils.MathHelper

class BernoulliNB {
    var data: Array<DataRow>
    var labels: Array<String>
    var labelSet: HashSet<String>
    var priors: HashMap<String, Double>
    var model: HashMap<String, HashMap<String, Double>>
    var numRows: Int = 0
    var distinctFeatures: HashSet<String>

    init {
        this.data = arrayOf()
        this.labels = arrayOf()
        this.labelSet = hashSetOf()
        this.model = hashMapOf()
        this.priors = hashMapOf()
        this.distinctFeatures = hashSetOf()
    }

    /**
     *  Fit model to given data and labels
     */
    fun fit(data: Array<DataRow>, labels: Array<String>) {
        this.data = data
        this.labels = labels
        for (label in labels) {
            labelSet.add(label)
        }
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
     *  Trains our model using the Bernouilli naive bayes formula
     *  We need to calculate number of documents/rows that a feature value appears in
     *  For more info: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes
     */
    private fun trainModel() : HashMap<String, HashMap<String, Double>> {
        val res = HashMap<String, HashMap<String, Double>>()
        val classSeparatedData = separateByClass()

        for (classVal in classSeparatedData.keys) {
            // Get all docs for a class
            val classData = classSeparatedData.get(classVal)

            // Calculate Priors
            this.priors.put(classVal, classData!!.size.toDouble()/this.numRows)

            // Get Occurrences
            val classFeatureOccurrences = aggregateOccurrencesPerClass(classData)

            // Calculate percentage of docs that a feature appears in per class
            val classProbabilities = hashMapOf<String, Double>()
            for (featureName in classFeatureOccurrences.keys) {
                // Use laplacian smoothing to avoid probabilities with 0 or 1
                classProbabilities.put(featureName, (classFeatureOccurrences.getValue(featureName) + 1) / (classData.size + 2))
            }
            res.put(classVal, classProbabilities)
        }
        return res
    }

    /**
     *  Make prediction of label based on multivariate Bernoulli event model, and take the MAP estimate
     *  The model can be found here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes
     */
    private fun predict(document: DataRow) : String {
        var bestProb = Int.MIN_VALUE.toDouble()
        var bestLabel = ""

        for (classVal in this.labelSet) {
            var likelihood = 0.0
            for (feature in document.keys) {
                val featureCount = document.get(feature) as Double
                var x_i = 0.0
                if (featureCount > 0) x_i = 1.0

                val p_ki = this.model.get(classVal)!!.get(feature)!!
                likelihood += (x_i * Math.log(p_ki)) + ((1 - x_i) * Math.log(1 - p_ki))
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
     *  Count number of documents that contain a feature
     */
    private fun aggregateOccurrencesPerClass(classData: MutableList<DataRow>) : HashMap<String, Double> {
        val res = HashMap<String, Double>()
        for (row in classData) {
            // Go through each document
            for(featureName in row.keys) {
                val count = row.get(featureName) as Double
                var presence =  0.0
                if (count > 0) presence = 1.0
                res.put(featureName, res.getOrDefault(featureName, 0.0) + presence)
                distinctFeatures.add(featureName)
            }
        }
        return res
    }

    /**
     *  Separates data into a mapping of class value to datarows belonging to that class
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
}
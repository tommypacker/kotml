package NaiveBayes

import Utils.DataRow
import Utils.MathHelper

class BernoulliNB {
    var data: MutableList<DataRow>
    var labels: MutableList<String>
    var labelSet: HashSet<String>
    var priors: HashMap<String, Double>
    var model: HashMap<String, HashMap<String, Double>>
    var numRows: Int = 0
    var distinctFeatures: HashSet<String>

    init {
        this.data = mutableListOf()
        this.labels = mutableListOf()
        this.labelSet = hashSetOf()
        this.model = hashMapOf()
        this.priors = hashMapOf()
        this.distinctFeatures = hashSetOf()
    }

    fun fit(data: MutableList<DataRow>, labels: MutableList<String>) {
        this.data = data
        this.labels = labels
        for (label in labels) {
            labelSet.add(label)
        }
        this.numRows = data.size
        this.model = trainModel()
    }

    fun getPredictions(testData: MutableList<DataRow>) : MutableList<String> {
        val res = mutableListOf<String>()
        for (row in testData) {
            res.add(predict(row))
        }
        return res
    }

    fun test(testData: MutableList<DataRow>, testLabels: MutableList<String>) : Double {
        val predictions = getPredictions(testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        return accuracy
    }

    private fun trainModel() : HashMap<String, HashMap<String, Double>> {
        val res = HashMap<String, HashMap<String, Double>>()
        val sepClassData = separateByLabels()

        for (classVal in sepClassData.keys) {
            // Get all docs for a class
            val classData = sepClassData.get(classVal)

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

    private fun predict(document: DataRow) : String {
        var bestProb = Int.MIN_VALUE.toDouble()
        var bestLabel = ""

        for (labelVal in this.labelSet) {
            var likelihood = 0.0
            for (feature in document.keys) {
                val featureCount = document.get(feature) as Double
                var x_i = 0.0
                if (featureCount > 0) x_i = 1.0

                val p_ki = this.model.get(labelVal)!!.get(feature)!!
                likelihood += (x_i * Math.log(p_ki)) + ((1 - x_i) * Math.log(1 - p_ki))
            }
            likelihood += Math.log(this.priors.get(labelVal)!!)
            if (likelihood > bestProb) {
                bestProb = likelihood
                bestLabel = labelVal
            }
        }
        return bestLabel
    }

    // Count number of documents that contain a feature
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

    private fun separateByLabels() : HashMap<String, MutableList<DataRow>> {
        val res = HashMap<String, MutableList<DataRow>>()
        for (i in 0..this.data.size-1) {
            val row = this.data.get(i)
            val labelVal = this.labels.get(i)
            if (!res.containsKey(labelVal)) {
                res.put(labelVal, mutableListOf())
            }
            res.get(labelVal)?.add(row)
        }
        return res
    }
}
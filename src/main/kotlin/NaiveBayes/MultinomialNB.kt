package NaiveBayes

import Utils.DataRow
import Utils.MathHelper

class MultinomialNB {
    var data: MutableList<DataRow>
    var labels: MutableList<String>
    var priors: HashMap<String, Double>
    var model: HashMap<String, HashMap<String, Double>>
    var numRows: Int = 0
    var distinctWords: HashSet<String>

    init {
        this.data = mutableListOf()
        this.labels = mutableListOf()
        this.model = hashMapOf()
        this.priors = hashMapOf()
        this.distinctWords = hashSetOf()
    }

    fun fit(data: MutableList<DataRow>, labels: MutableList<String>) {
        this.data = data
        this.labels = labels
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

        for (labelVal in sepClassData.keys) {
            // Get all docs for a class
            val classData = sepClassData.get(labelVal)
            // Calculate Priors
            this.priors.put(labelVal, classData!!.size.toDouble()/this.numRows)
            val aggregatedData = aggregateCountsPerClass(classData)
            //println(labelVal + ": " + aggregatedData)
            val classProbabilties = HashMap<String, Double>()

            var totalNumWords = 0
            for (featureName in aggregatedData.keys) {
                totalNumWords += aggregatedData.get(featureName)!!
            }
            val n = distinctWords.size
            for (featureName in aggregatedData.keys) {
                // Use LaPlace smoothing
                classProbabilties.put(featureName, (aggregatedData.get(featureName)!!.toDouble() + 1) / (totalNumWords + n))
            }
            res.put(labelVal, classProbabilties)
        }
        return res
    }

    private fun predict(document: DataRow) : String {
        var bestProb = Double.MIN_VALUE
        var bestLabel = ""
        for (labelVal in this.labels) {
            var curProb = 1.0
            var likelihood = 1.0
            var totalWords = 0
            for (word in document.keys) {
                totalWords += (document.get(word) as Double).toInt()
            }

            for (word in document.keys) {
                val wordCount = document.get(word) as Double
                likelihood *= Math.pow(this.model.get(labelVal)!!.get(word)!!, wordCount)
            }
            curProb *= likelihood
            curProb *= this.priors.get(labelVal)!!
            if (curProb > bestProb) {
                bestProb = curProb
                bestLabel = labelVal
            }
        }
        return bestLabel
    }

    private fun aggregateCountsPerClass(classData: MutableList<DataRow>) : HashMap<String, Int> {
        val res = HashMap<String, Int>()
        for (row in classData) {
            for(featureName in row.keys) {
                val count = (row.get(featureName) as Double).toInt()
                res.put(featureName, res.getOrDefault(featureName, 0) + count)
                distinctWords.add(featureName)
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
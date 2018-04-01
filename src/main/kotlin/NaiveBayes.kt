import Utils.MathHelper

class GaussianNB(data: MutableList<Map<String, Any?>>, labels: MutableList<String>) {

    var data: MutableList<Map<String, Any?>>
    var labels: MutableList<String>
    var model: HashMap<String, HashMap<String, Pair<Double, Double>>>

    init {
        this.data = data
        this.labels = labels
        this.model = HashMap()
    }

    /**
     * Fit our model to the dataset
     */
    fun fit() {
        this.model = trainModel()
    }

    fun test(testData: MutableList<Map<String, Any?>>, testLabels: MutableList<String>) : Double {
        val predictions = getPredictions(model, testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        return accuracy
    }

    fun trainModel() : HashMap<String, HashMap<String, Pair<Double, Double>>> {
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

    fun separateByLabels() : HashMap<String, MutableList<Map<String, Any?>>> {
        val res = HashMap<String, MutableList<Map<String, Any?>>>()
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

    fun separateFeatures(dataCols: MutableList<Map<String, Any?>>) : HashMap<String, DoubleArray> {
        val featureMap = HashMap<String, DoubleArray>()
        val cols = this.data.get(0).keys
        for (colName in cols) {
            val colVals = DoubleArray(dataCols.size)

            // Need ability to slice by index rather than manually doing this
            for (i in 0..dataCols.size-1) {
                var curRow = dataCols.get(i).get(colName)
                if (curRow is Int) {
                    curRow = curRow.toDouble()
                } else if (curRow is String) {
                    curRow = 0.1
                } else {
                    curRow = curRow as Double
                }
                colVals[i] = curRow
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

    fun calculateClassProbabilities(summaries: HashMap<String, HashMap<String, Pair<Double, Double>>>,
                                    inputVector: Map<String, Any?>) : HashMap<String, Double> {
        val res = HashMap<String, Double>()
        for (labelVal in summaries.keys) {
            var classProbability = 1.0
            val labelFeatureSummary = summaries.get(labelVal)
            for (featureName in labelFeatureSummary!!.keys) {
                val summary = labelFeatureSummary.get(featureName)
                val mean = summary!!.first
                val stdev = summary.second
                var curFeatureItem = inputVector.get(featureName)
                if (curFeatureItem is Int) {
                    curFeatureItem = curFeatureItem.toDouble()
                } else if (curFeatureItem is String) {
                    curFeatureItem = 0.1
                } else {
                    curFeatureItem = curFeatureItem as Double
                }
                val x = curFeatureItem
                classProbability *= MathHelper.calculateProbability(x, mean, stdev)
            }
            //println("cp: " + classProbability)
            res.put(labelVal, classProbability)
        }
        return res
    }

    fun predict(summaries: HashMap<String, HashMap<String, Pair<Double, Double>>>,
                inputVector: Map<String, Any?>) : String {
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
                       testData: MutableList<Map<String, Any?>>) : MutableList<String> {
        var res = mutableListOf<String>()
        val numRows = testData.size
        for (i in 0..numRows-1) {
            val prediction = predict(summaries, testData.get(i))
            res.add(prediction)
        }
        return res
    }
}
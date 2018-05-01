package kotml.knn

import kotml.utils.DataRow
import kotml.utils.MathHelper

class KNN (val data: Array<DataRow>, val labels: Array<String>, val k: Int = 1, val weighted: Boolean = false){

    /**
     *  Test our model by making predictions and comparing them to the actual labels
     */
    fun test(testData: Array<DataRow>, testLabels: Array<String>): Double {
        val predictions = getPredictions(testData)
        val accuracy = MathHelper.getAccuracy(testLabels, predictions)
        return accuracy
    }

    /**
     *  Gets an Array of label predictions for given test data
     */
    fun getPredictions(testData: Array<DataRow>): Array<String> {
        val res = mutableListOf<String>()
        for (testRow in testData) {
            res.add(predict(testRow))
        }
        return res.toTypedArray()
    }

    /**
     *  Predicts the label of a DataRow using the K-Nearest Neighbor algorithm.
     *  It sorts all the distances and then returns the label with the most neighbors.
     */
    fun predict(inputRow: DataRow): String {
        // Calculate distances
        val distances = mutableListOf<Pair<Double, String>>()
        for (i in 0..data.size - 1) {
            val dataRow = data[i]
            val label = labels[i]
            val distance = euclideanDistance(inputRow, dataRow)
            distances.add(Pair(distance, label))
        }

        // Sort distances
        distances.sortBy { it.first }

        // Aggregate neighbor value/counts
        val neighborCounts = hashMapOf<String, Double>()
        if (!weighted) {
            val neighbors = distances.slice(0..k-1)
            for (neighbor in neighbors) {
                val neighborLabel = neighbor.second
                neighborCounts.put(neighborLabel, neighborCounts.getOrDefault(neighborLabel, 0.0) + 1.0)
            }
        } else {
            for (distancePair in distances) {
                val neighborDistance = 1 / distancePair.first
                val neighborLabel = distancePair.second
                neighborCounts.put(neighborLabel, neighborCounts.getOrDefault(neighborLabel, 0.0) + neighborDistance)
            }
        }

        // Find most common or highest weighted neighbor label
        val ranks = neighborCounts.toList().sortedBy { (_, value) -> value }.reversed()
        return ranks[0].first
    }

    /**
     *  Calculates the Euclidean distance between two DataRows
     */
    fun euclideanDistance(A: DataRow, B: DataRow): Double {
        var distance = 0.0
        for (key in A.keys) {
            val aVal = A.get(key) as Double
            val bVal = B.get(key) as Double
            distance += Math.pow((aVal - bVal), 2.0)
        }
        return distance
    }
}
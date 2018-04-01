package Utils

class MathHelper {
    companion object {
        fun stdev(vals: DoubleArray): Double {
            val avg = vals.average()
            val varianceVals = vals.map { i: Double -> Math.pow(i - avg, 2.0) / (vals.size - 1) }
            val variance = varianceVals.sum()
            return Math.sqrt(variance)
        }

        fun calculateProbability(x: Double, mean: Double, stdev: Double) : Double {
            val variance = Math.pow(stdev, 2.0)
            val exponent = -(Math.pow(x - mean, 2.0) / (2 * variance))
            val eRaised = Math.pow(Math.E, exponent)
            return (1 / Math.sqrt(2 * Math.PI * variance)) * eRaised
        }

        fun getAccuracy(testLabels: MutableList<String>, predictions: MutableList<String>) : Double {
            var correct = 0.0
            val numRows = testLabels.size
            for (i in 0..numRows-1) {
                //println(predictions.get(i))
                //println(testLabels.get(i))
                //println()
                if (predictions.get(i) == (testLabels.get(i))) {
                    correct += 1
                }
            }
            return (correct / numRows) * 100.0
        }
    }
}
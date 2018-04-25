package kotml.Utils

class MathHelper {
    companion object {
        fun stdev(vals: DoubleArray): Double {
            val avg = vals.average()
            val varianceVals = vals.map { i: Double -> Math.pow(i - avg, 2.0) / (vals.size - 1) }
            val variance = varianceVals.sum()
            return Math.sqrt(variance)
        }

        fun calculateGaussian(x: Double, mean: Double, stdev: Double) : Double {
            val variance = Math.pow(stdev, 2.0)
            val exponent = -(Math.pow(x - mean, 2.0) / (2 * variance))
            val eRaised = Math.pow(Math.E, exponent)
            return (1 / Math.sqrt(2 * Math.PI * variance)) * eRaised
        }

        fun factorial(x: Int) : Int {
            var res = 1
            for (i in 1..x) res *= i
            return res
        }

        fun getAccuracy(testLabels: Array<String>, predictions: Array<String>) : Double {
            var correct = 0
            val numRows = testLabels.size
            for (i in 0..numRows-1) {
                if (predictions.get(i) == (testLabels.get(i))) {
                    correct += 1
                }
            }
            return (correct.toDouble() / numRows) * 100.0
        }
    }
}
package kotml.utils

class MathHelper {
    companion object {

        /**
         * Calculates the likelihood of x given the mean and stdev of a dataset
         * using an assumed Gaussian distribution
         */
        fun calculateGaussian(x: Double, mean: Double, stdev: Double) : Double {
            val variance = Math.pow(stdev, 2.0)
            val exponent = -(Math.pow(x - mean, 2.0) / (2 * variance))
            val eRaised = Math.pow(Math.E, exponent)
            return (1 / Math.sqrt(2 * Math.PI * variance)) * eRaised
        }

        /**
         * Calculates the accuracy of a list of predictions given the actual labels
         */
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
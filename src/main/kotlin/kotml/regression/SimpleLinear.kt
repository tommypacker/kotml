package kotml.regression

import koma.create
import koma.extensions.map

class SimpleLinear {
    var slope = 1.0
    var bias = 0.0

    fun train(X: DoubleArray, y: DoubleArray, learningRate: Double, epochs: Int) {
        val costs = mutableListOf<Double>()

        for (k in 1..epochs) {
            updateWeights(X, y, learningRate)
            costs.add(mseCostFunction(X, y))
        }

        println("Final Slope: " + slope)
    }

    fun updateWeights(X: DoubleArray, y: DoubleArray, learningRate: Double) {
        val N = X.size
        val hypothesis = create(X) * slope
        val loss = hypothesis - create(y)

        val gradient = (create(X) * loss.transpose()).elementSum() / N
        slope -= (gradient * learningRate)
    }

    fun mseCostFunction(X: DoubleArray, y: DoubleArray): Double {
        val N = X.size
        val hypothesis = create(X) * slope
        val loss = hypothesis - create(y)
        val cost = loss.map { Math.pow(it, 2.0) }.elementSum() / N
        return cost
    }
}
package kotml.regression

import koma.create
import koma.extensions.get
import koma.extensions.map
import koma.eye
import koma.matrix.Matrix
import koma.zeros

class Linear {
    /**
     *  Implemented from https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
     */

    var theta = eye(1)

    fun predict(xVec: DoubleArray): Double {
        val x = create(xVec)
        return (x * theta).elementSum()
    }

    fun train(X: Matrix<Double>, yVals: DoubleArray, learningRate: Double, epochs: Int) {
        theta = zeros(X.numCols(), 1)
        val y = create(yVals)
        val costs = mutableListOf<Double>()
        for (k in 1..epochs) {
            updateWeights(X, y, learningRate)
            costs.add(mseCostFunction(X, y))
        }
        println("Final theta values: " + theta)
    }

    fun updateWeights(X: Matrix<Double>, y: Matrix<Double>, learningRate: Double) {
        val N = X.numRows()
        val hypothesis = X * theta
        val loss = hypothesis - y.transpose()

        val gradient = (X.transpose() * loss).map { it / N * learningRate }
        theta = theta - gradient
    }

    fun mseCostFunction(X: Matrix<Double>, y: Matrix<Double>): Double {
        val N = X.numRows()
        val hypothesis = X * theta
        val loss = hypothesis - y.transpose()
        val cost = loss.map { Math.pow(it, 2.0) }.elementSum() / N
        return cost
    }
}
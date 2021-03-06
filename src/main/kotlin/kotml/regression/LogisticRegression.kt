package kotml.regression

import koma.*
import koma.extensions.map
import koma.matrix.Matrix

class LogisticRegression {

    /**
     *  Implemented from https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
     */

    var theta = eye(0)
    val costs = mutableListOf<Double>()


    /**
     *  Predict the values of a matrix of values using our linear regression model
     */
    fun predictValues(features: Matrix<Double>): Matrix<Double> {
        return addBiasCol(features) * theta
    }

    /**
     *  Train our regression model using gradient descent to minimize the mean squared error
     */
    fun train(features: Matrix<Double>, yVals: DoubleArray, learningRate: Double, epochs: Int) {
        val X = addBiasCol(features)
        theta = zeros(X.numCols(), 1)
        val y = create(yVals)

        for (k in 1..epochs) {
            updateWeights(X, y, learningRate)
        }
        println("Final theta values: " + theta)
    }

    /**
     *  Update the weights of our model using gradient descent
     */
    fun updateWeights(X: Matrix<Double>, y: Matrix<Double>, learningRate: Double) {
        val N = X.numRows()
        val hypothesis = (X * theta).map { sigmoid(it) }
        val loss = hypothesis - y.transpose()
        costs.add(calculateCost(y, hypothesis))

        val gradient = (X.transpose() * loss).map { it / N * learningRate }
        theta = theta - gradient
    }

    /**
     *  Find the current cost of our model using Cross Entropy
     */
    fun calculateCost(y: Matrix<Double>, predictions: Matrix<Double>): Double {
        val m = y.numRows()
        val classOneCosts = -y * (ln(predictions))
        val classTwoCosts = -(y.map { 1 - it }) * ln(predictions.map { 1 - it })
        val costs = classOneCosts - classTwoCosts
        return costs.elementSum() / m
    }

    /**
     *  Add a column of 1s at the front of our feature matrix to account for biases
     */
    fun addBiasCol(X: Matrix<Double>): Matrix<Double> {
        val m = X.numRows()
        val n = X.numCols()
        val Xn = ones(m, n + 1)
        for (i in 0..X.numCols()-1) {
            Xn.setCol(i+1, X.getCol(i))
        }
        return Xn
    }

    fun sigmoid(z: Double): Double {
        return 1 / (1 + Math.exp(z))
    }
}
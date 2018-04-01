import krangl.DataFrame
import krangl.readCSV

fun main(args: Array<String>) {
    val data = DataFrame.readCSV("src/main/kotlin/iris.csv")
    var NB = GaussianNB(data.select("sepal_length", "sepal_width", "petal_length", "petal_width"), data.select("species"))
    NB.splitDataset(0.67)
    NB.fit()
}
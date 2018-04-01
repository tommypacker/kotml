import krangl.DataFrame
import krangl.readCSV

fun main(args: Array<String>) {
    val data = DataFrame.readCSV("src/main/kotlin/iris.csv")
    var NB = GaussianNB(data)
    NB.fit()
}
package Utils

import krangl.DataFrame
import krangl.DataFrameRow

class DataTransformer {
    companion object {

        /**
         * Converts a DataFrame into a list of data rows (for features) and strings (labels)
         */
        fun transformDataframe(rawData: DataFrame, data: MutableList<DataFrameRow>, labels: MutableList<String>) {
            val numCols = rawData.ncol
            val labelName = rawData.cols.get(numCols-1).name

            for (i in 0..rawData.nrow-1) {
                val curRow = rawData.row(i)
                labels.add(curRow.get(labelName) as String)
                data.add(curRow.minus(labelName))
            }
        }
    }
}
import org.apache.spark.mllib.linalg.DenseMatrix

object MatrixMultiplBySpark {
  def main(args: Array[String]) {

    val random = new scala.util.Random

    def random2dArray(dim1: Int, dim2: Int, maxValue: Int): DenseMatrix = {
      new DenseMatrix(dim1, dim2, Array.fill(dim1 * dim2)(random.nextDouble() * maxValue))
    }

    val no_of_matrix_multiplications = 1000 
    val matrix_dimensions = 1000   

    val array_left = random2dArray(matrix_dimensions, matrix_dimensions, 100)
    val array_right = random2dArray(matrix_dimensions, matrix_dimensions, 100)

    var result = array_left.multiply(array_right)

    val start = System.currentTimeMillis()
    for (_ <- 1 to no_of_matrix_multiplications) {
      result = array_left.multiply(array_right)
    }
    val end = System.currentTimeMillis()

    println(s"\nTime for $no_of_matrix_multiplications runs: ${(end - start) / 1000.0} s\n")
    println(s"Average time per run: ${((end - start) / 1000.0) / no_of_matrix_multiplications} s \n")
  }
}

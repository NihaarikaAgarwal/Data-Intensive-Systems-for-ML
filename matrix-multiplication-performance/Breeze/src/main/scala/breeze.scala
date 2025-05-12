import breeze.linalg._

object BreezeMatrixMultiply {
  def main(args: Array[String]): Unit = {
    val matrix_dimension = 1000
    val no_of_matrix_multiplication = 100
    
    val array_left = DenseMatrix.rand[Double](1000, 1000)
    val array_right = DenseMatrix.rand[Double](1000, 1000)

    var result = array_left * array_right

    val start = System.currentTimeMillis()

    for (_ <- 1 to no_of_matrix_multiplication)
      result = array_left * array_right
    val end = System.currentTimeMillis()
    
    println(s"\nTime for $no_of_matrix_multiplication runs: ${(end - start) / 1000.0} s\n")
    println(s"Average time per run: ${((end - start) / 1000.0) / no_of_matrix_multiplication} s \n")
  }
}

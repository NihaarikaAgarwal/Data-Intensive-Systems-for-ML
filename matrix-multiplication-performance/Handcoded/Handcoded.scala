object MatrixMultiplyByHandcoded {
  def main(args: Array[String]): Unit = {

    val random = new scala.util.Random

    def making_random2Darray(dim1: Int, dim2: Int, maxValue: Int): Array[Array[Double]] = 
      Array.fill(dim1, dim2) {
        1.0 + random.nextDouble() * maxValue
      }

    val no_of_matrix_multiplication = 100
    val n = 1000 // Array dimension
    val array_left = making_random2Darray(n, n, 100)
    val array_right = making_random2Darray(n, n, 100)
    var result = Array.ofDim[Double](n, n)

    def multiply(array_left: Array[Array[Double]], array_right: Array[Array[Double]]): Unit = {
      for (i <- array_left.indices)
        for (j <- array_right(0).indices)
          for (k <- array_right.indices)
            result(i)(j) += array_left(i)(k) * array_right(k)(j)
    }

    val start = System.currentTimeMillis()
    for (_ <- 1 to no_of_matrix_multiplication)
      multiply(array_left, array_right)
    val end = System.currentTimeMillis()

    println(s"\nTime for $no_of_matrix_multiplication runs: ${(end - start) / 1000.0} s \n")
	println(s"Average time per run: ${((end - start) / 1000.0) / n} s \n")

  }
}

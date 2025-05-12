name := "BreezeMatrixMultiply"  // Project name

version := "0.1"  // Version of your project

scalaVersion := "2.12.18"  // Scala version that is compatible with Breeze

// Breeze dependency for matrix operations
libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.scalanlp" %% "breeze" % "0.13",  
  "org.apache.spark" %% "spark-core" % "3.4.0", 
  "org.apache.spark" %% "spark-sql"  % "3.4.0"   
)

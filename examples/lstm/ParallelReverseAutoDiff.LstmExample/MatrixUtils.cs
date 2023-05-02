namespace ParallelReverseAutoDiff.LstmExample
{
    public static class MatrixUtils
    {
        public static double[][] InitializeZeroMatrix(int numRows, int numCols)
        {
            double[][] matrix = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                matrix[i] = new double[numCols];
            }
            return matrix;
        }

        public static double[][][] InitializeZeroMatrix(int numLayers, int numRows, int numCols)
        {
            double[][][] matrix = new double[numLayers][][];
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
            {
                matrix[layerIndex] = new double[numRows][];
                for (int i = 0; i < numRows; i++)
                {
                    matrix[layerIndex][i] = new double[numCols];
                }
            }
            return matrix;
        }

        public static double[][][][] InitializeZeroMatrix(int numTimeSteps, int numLayers, int numRows, int numCols)
        {
            double[][][][] m = new double[numTimeSteps][][][];
            for (int t = 0; t < numTimeSteps; ++t)
            {
                double[][][] matrix = new double[numLayers][][];
                for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
                {
                    matrix[layerIndex] = new double[numRows][];
                    for (int i = 0; i < numRows; i++)
                    {
                        matrix[layerIndex][i] = new double[numCols];
                    }
                }
                m[t] = matrix;
            }
            return m;
        }

        public static double[][] ClipGradients(double[][] gradients, double clipValue, double minValue = 1E-6)
        {
            var standardizedMatrix = StandardizedMatrix(gradients);
            int numRows = gradients.Length;
            int numCols = gradients[0].Length;
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    var value = Math.Min(clipValue, 1 + Math.Abs(standardizedMatrix[i][j]));
                    // Clip the gradient values
                    if (gradients[i][j] > value)
                    {
                        gradients[i][j] = value;
                    }
                    else if (gradients[i][j] < -value)
                    {
                        gradients[i][j] = -value;
                    }

                    // Apply the minimum threshold value
                    if (gradients[i][j] > 0 && gradients[i][j] < minValue)
                    {
                        gradients[i][j] = minValue;
                    }
                    else if (gradients[i][j] < 0 && gradients[i][j] > -minValue)
                    {
                        gradients[i][j] = -minValue;
                    }
                }
            }

            return gradients;
        }

        public static double[][][] ClipGradients(double[][][] gradients, double clipValue, double minValue = 1E-6)
        {
            int numMatrices = gradients.Length;

            for (int k = 0; k < numMatrices; k++)
            {
                int numRows = gradients[k].Length;
                int numCols = gradients[k][0].Length;
                var standardizedMatrix = StandardizedMatrix(gradients[k]);

                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        var value = Math.Min(clipValue, 1 + Math.Abs(standardizedMatrix[i][j]));
                        // Clip the gradient values
                        if (gradients[k][i][j] > value)
                        {
                            gradients[k][i][j] = value;
                        }
                        else if (gradients[k][i][j] < -value)
                        {
                            gradients[k][i][j] = -value;
                        }

                        // Apply the minimum threshold value
                        if (gradients[k][i][j] > 0 && gradients[k][i][j] < minValue)
                        {
                            gradients[k][i][j] = minValue;
                        }
                        else if (gradients[k][i][j] < 0 && gradients[k][i][j] > -minValue)
                        {
                            gradients[k][i][j] = -minValue;
                        }
                    }
                }
            }

            return gradients;
        }

        public static double[][] StandardizedMatrix(double[][] matrix)
        {
            // Calculate the mean
            double sum = 0;
            int count = 0;
            foreach (double[] row in matrix)
            {
                foreach (double value in row)
                {
                    sum += value;
                    count++;
                }
            }
            double mean = sum / count;

            // Calculate the standard deviation
            double varianceSum = 0;
            foreach (double[] row in matrix)
            {
                foreach (double value in row)
                {
                    varianceSum += Math.Pow(value - mean, 2);
                }
            }
            double stdDev = Math.Sqrt(varianceSum / count);

            // Calculate the standardized matrix
            double[][] standardizedMatrix = new double[matrix.Length][];
            for (int i = 0; i < matrix.Length; i++)
            {
                standardizedMatrix[i] = new double[matrix[i].Length];
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    standardizedMatrix[i][j] = (matrix[i][j] - mean) / stdDev;
                }
            }

            return standardizedMatrix;
        }

        public static double[][] To2DArray(double[][][] matrices)
        {
            int numMatrices = matrices.Length;
            double[][] matrix = new double[numMatrices][];
            for (int i = 0; i < numMatrices; ++i)
            {
                matrix[i] = matrices[i][0];
            }
            return matrix;
        }

        public static double[] To1DArray(double[][][] matrices)
        {
            int numMatrices = matrices.Length;
            double[] array = new double[numMatrices];
            for (int i = 0; i < numMatrices; ++i)
            {
                array[i] = matrices[i][0][0];
            }
            return array;
        }

        public static void SetInPlace(double[][][] matrices, double[][][] value)
        {
            int numMatrices = matrices.Length;
            int numRows = matrices[0].Length;
            int numCols = matrices[0][0].Length;
            for (int i = 0; i < numMatrices; ++i)
            {
                for (int j = 0; j < numRows; ++j)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        matrices[i][j][k] = value[i][j][k];
                    }
                }
            }
        }

        public static void ClearArrays4D(double[][][][][] matrices)
        {
            int numMatrices = matrices.Length;
            int numTimesteps = matrices[0].Length;
            int numLayers = matrices[0][0].Length;
            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                for (int j = 0; j < numTimesteps; ++j)
                {
                    for (int k = 0; k < numLayers; ++k)
                    {
                        int numRows = matrices[i][j][k].Length;
                        int numCols = matrices[i][j][k][0].Length;
                        for (int l = 0; l < numRows; ++l)
                        {
                            matrices[i][j][k][l] = new double[numCols];
                        }
                    }
                }
            });
        }

        public static void ClearArrays3D(double[][][][] matrices)
        {
            int numMatrices = matrices.Length;
            int numLayers = matrices[0].Length;
            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                for (int j = 0; j < numLayers; ++j)
                {
                    int numRows = matrices[i][j].Length;
                    int numCols = matrices[i][j][0].Length;
                    for (int k = 0; k < numRows; ++k)
                    {
                        matrices[i][j][k] = new double[numCols];
                    }
                }
            });
        }

        public static void ClearArrays2D(double[][][] matrices)
        {
            int numMatrices = matrices.Length;
            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                int numRows = matrices[i].Length;
                int numCols = matrices[i][0].Length;
                for (int j = 0; j < numRows; ++j)
                {
                    matrices[i][j] = new double[numCols];
                }
            });
        }

        public static double FrobeniusNorm(double[][] weightMatrix)
        {
            double sum = 0.0;

            for (int i = 0; i < weightMatrix.Length; i++)
            {
                for (int j = 0; j < weightMatrix[i].Length; j++)
                {
                    sum += weightMatrix[i][j] * weightMatrix[i][j];
                }
            }

            return Math.Sqrt(sum);
        }

        public static double LearningRateReductionFactor(double frobeniusNorm, double maxNorm, double minFactor)
        {
            if (frobeniusNorm <= maxNorm)
            {
                return 1.0;
            }
            else
            {
                double reductionFactor = Math.Pow(maxNorm / frobeniusNorm, 3);
                return Math.Max(reductionFactor, minFactor);
            }
        }

        public static double[][] MatrixAdd(double[][] a, double[][] b, double[][] c)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j] + c[i][j];
                }
            }
            return result;
        }

        public static double[][] HadamardProduct(double[][] matrixA, double[][] matrixB)
        {
            // Check if the dimensions of the matrices match
            int rows = matrixA.Length;
            int cols = matrixA[0].Length;
            if (rows != matrixB.Length || cols != matrixB[0].Length)
            {
                throw new ArgumentException("Matrices must have the same dimensions.");
            }

            // Perform element-wise multiplication
            var result = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            }

            return result;
        }

        public static double[][] MatrixAdd(double[][] a, double[][] b, double[][] c, double[][] d)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j] + c[i][j] + d[i][j];
                }
            }
            return result;
        }

        public static double[][] MatrixAdd(double[][] a, double[][] b)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j];
                }
            }
            return result;
        }

        public static double[][] MatrixAdd(double[][] matrix, double[] vector)
        {
            int numRows = matrix.Length;
            int numCols = matrix[0].Length;
            double[][] result = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = matrix[i][j] + vector[j];
                }
            }

            return result;
        }

        public static double[][] ScalarMultiply(double scalar, double[][] matrix)
        {
            int numRows = matrix.Length;
            int numCols = matrix[0].Length;

            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = scalar * matrix[i][j];
                }
            }

            return result;
        }

        public static double[][] MatrixAdd(double[][] a, double[][] b, double[] c)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j] + c[j];
                }
            }
            return result;
        }

        public static double[][] InitializeRandomMatrixWithXavierInitialization(int numRows, int numCols)
        {
            double[][] matrix = new double[numRows][];
            var localRandom = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));

            Parallel.For(0, numRows, i =>
            {
                matrix[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    matrix[i][j] = (localRandom.Value.NextDouble() * 2 - 1) * Math.Sqrt(6.0 / (numRows + numCols));
                }
            });

            return matrix;
        }

        public static double[][][] InitializeRandomMatrixWithXavierInitialization(int numLayers, int numRows, int numCols)
        {
            double[][][] matrix = new double[numLayers][][];

            var localRandom = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));

            Parallel.For(0, numLayers, layerIndex =>
            {
                matrix[layerIndex] = new double[numRows][];
                for (int i = 0; i < numRows; i++)
                {
                    matrix[layerIndex][i] = new double[numCols];
                    for (int j = 0; j < numCols; j++)
                    {
                        matrix[layerIndex][i][j] = (localRandom.Value.NextDouble() * 2 - 1) * Math.Sqrt(6.0 / (numRows + numCols));
                    }
                }
            });

            return matrix;
        }
    }
}

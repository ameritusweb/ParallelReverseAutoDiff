//------------------------------------------------------------------------------
// <copyright file="Tensor.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;
    using MKLNET;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A flat tensor.
    /// </summary>
    public partial class Tensor
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class.
        /// </summary>
        /// <param name="shape">The dimensions of the tensor.</param>
        public Tensor(int[] shape)
        {
            this.Shape = shape;
            this.Data = new double[this.GetTotalSize(shape)];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class.
        /// </summary>
        /// <param name="original">The original tensor.</param>
        public Tensor(Tensor original)
        {
            this.Shape = new int[original.Shape.Length];
            Array.Copy(original.Shape, this.Shape, original.Shape.Length);
            this.Data = new double[original.Data.Length];
            Buffer.BlockCopy(original.Data, 0, this.Data, 0, original.Data.Length * sizeof(double));
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="data">The data.</param>
        /// <exception cref="ArgumentException">The data length does not match.</exception>
        public Tensor(int[] shape, double[] data)
        {
            if (this.GetTotalSize(shape) != data.Length)
            {
                throw new ArgumentException("Data length does not match shape size.");
            }

            this.Shape = shape;
            this.Data = data;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class from a 2D array.
        /// </summary>
        /// <param name="array">The 2D array to initialize the tensor with.</param>
        public Tensor(double[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);

            this.Shape = new int[] { rows, cols };
            this.Data = new double[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.Data[(i * cols) + j] = array[i, j];
                }
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class with a single value repeated for all elements.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="value">The value to fill the tensor with.</param>
        public Tensor(int[] shape, double value)
        {
            this.Shape = shape;
            int totalSize = shape.Aggregate(1, (a, b) => a * b);
            this.Data = new double[totalSize];
            Array.Fill(this.Data, value);
        }

        /// <summary>
        /// Gets the data for the tensor.
        /// </summary>
        public double[] Data { get; private set; }

        /// <summary>
        /// An indexer.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>The data value.</returns>
        public double this[params int[] indices]
        {
            get => this.Data[this.GetIndex(indices)];
            set => this.Data[this.GetIndex(indices)] = value;
        }

        /// <summary>
        /// Returns 1 if a tensor value is greater than a scalar, 0 otherwise.
        /// </summary>
        /// <param name="a">The tensor to compare.</param>
        /// <param name="scalar">The scalar to compare.</param>
        /// <returns>The resultant tensor.</returns>
        public static Tensor operator >(Tensor a, double scalar)
        {
            var result = new Tensor(a.Shape);
            Parallel.For(0, a.Data.Length, i =>
            {
                result.Data[i] = a.Data[i] > scalar ? 1 : 0;
            });

            return result;
        }

        /// <summary>
        /// Returns 1 if a tensor value is less than a scalar, 0 otherwise.
        /// </summary>
        /// <param name="a">The tensor to compare.</param>
        /// <param name="scalar">The scalar to compare.</param>
        /// <returns>The resultant tensor.</returns>
        public static Tensor operator <(Tensor a, double scalar)
        {
            var result = new Tensor(a.Shape);
            Parallel.For(0, a.Data.Length, i =>
            {
                result.Data[i] = a.Data[i] < scalar ? 1 : 0;
            });

            return result;
        }

        /// <summary>
        /// Generates a tensor containing values from start to (but not including) limit with a specified step (delta).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="limit">The exclusive upper bound of the sequence.</param>
        /// <param name="delta">The step size between elements (default is 1).</param>
        /// <returns>A 1D tensor with evenly spaced values.</returns>
        public static Tensor Range(double start, double limit, double delta = 1.0)
        {
            if (delta == 0)
            {
                throw new ArgumentException("Delta (step size) cannot be zero.");
            }

            // Calculate number of elements
            int numElements = (int)Math.Ceiling((limit - start) / delta);

            if (numElements <= 0)
            {
                return new Tensor(new int[] { 0 });  // Return empty tensor if the range is invalid
            }

            // Allocate data for the result tensor
            var result = new Tensor(new int[] { numElements });

            Parallel.For(0, numElements, i =>
            {
                result.Data[i] = start + (i * delta);
            });

            return result;
        }

        /// <summary>
        /// Creates a stack of tensors along a specified axis.
        /// </summary>
        /// <param name="tensors">Tensors.</param>
        /// <param name="axis">The axis to stack on.</param>
        /// <returns>The new tensor.</returns>
        /// <exception cref="ArgumentException">Cannot be empty.</exception>
        public static Tensor Stack(Tensor[] tensors, int axis = 0)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException("The input list of tensors cannot be empty.");
            }

            // Ensure all tensors have the same shape
            var shape = tensors[0].Shape;
            foreach (var tensor in tensors)
            {
                if (!shape.SequenceEqual(tensor.Shape))
                {
                    throw new ArgumentException("All input tensors must have the same shape.");
                }
            }

            // Handle negative axis
            if (axis < 0)
            {
                axis = shape.Length + 1 + axis;
            }

            // Validate axis
            if (axis < 0 || axis > shape.Length)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Determine the output shape
            var outputShape = new int[shape.Length + 1];
            Array.Copy(shape, 0, outputShape, 0, axis);
            outputShape[axis] = tensors.Length;
            Array.Copy(shape, 0, outputShape, axis + 1, shape.Length - axis);

            var totalSize = tensors.Length * shape.Aggregate((a, b) => a * b);
            var outputData = new double[totalSize];
            var outputTensor = new Tensor(outputShape, outputData);

            var strides = new int[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            Parallel.For(0, tensors.Length, i =>
            {
                var tensorData = tensors[i].Data;
                var offset = i * tensorData.Length;
                Buffer.BlockCopy(tensorData, 0, outputData, offset * sizeof(double), tensorData.Length * sizeof(double));
            });

            return outputTensor;
        }

        /// <summary>
        /// Create a tensor array.
        /// </summary>
        /// <param name="count">The number of tensors.</param>
        /// <param name="shape">The shape of each tensor.</param>
        /// <returns>The resultant tensor array.</returns>
        public static Tensor[] ToTensorArray(int count, int[] shape)
        {
            var tensorArray = new Tensor[count];
            for (int i = 0; i < count; i++)
            {
                tensorArray[i] = new Tensor(shape);
            }

            return tensorArray;
        }

        /// <summary>
        /// Creates a flat array from the tensors along the specified indices.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>The flat array.</returns>
        public static Tensor CreateFlatArray(Tensor[] tensors, int[] indices)
        {
            int[] shape = tensors[0].Shape;
            double[] flatArray = new double[tensors.Length * indices.Length * shape[0]];
            int flatIndex = 0;

            foreach (var tensor in tensors)
            {
                foreach (var index in indices)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        flatArray[flatIndex++] = tensor[i, index];
                    }
                }
            }

            return new Tensor(new int[] { 1, flatArray.Length }, flatArray);
        }

        /// <summary>
        /// Create a tensor mask.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="conditionFunc">The condition.</param>
        /// <returns>The tensor mask.</returns>
        public static Tensor CreateMask(Tensor tensor, Func<double, bool> conditionFunc)
        {
            var maskData = new double[tensor.Data.Length];
            Parallel.For(0, tensor.Data.Length, i =>
            {
                maskData[i] = conditionFunc(tensor.Data[i]) ? 1.0 : 0.0;
            });

            return new Tensor(tensor.Shape, maskData);
        }

        /// <summary>
        /// Generates a tensor with random values drawn from a uniform distribution in the range [minValue, maxValue).
        /// </summary>
        /// <param name="shape">The shape of the output tensor.</param>
        /// <param name="minValue">The minimum value of the distribution (inclusive).</param>
        /// <param name="maxValue">The maximum value of the distribution (exclusive).</param>
        /// <returns>A tensor filled with random values in the specified range.</returns>
        /// <exception cref="ArgumentException">Thrown if maxValue is less than or equal to minValue.</exception>
        public static Tensor RandomUniform(int[] shape, double minValue = 0.0, double maxValue = 1.0)
        {
            if (maxValue <= minValue)
            {
                throw new ArgumentException("maxValue must be greater than minValue.");
            }

            // Calculate the number of elements in the tensor
            int totalSize = shape.Aggregate(1, (x, y) => x * y);

            // Allocate memory for tensor data
            var result = new Tensor(shape);

            // Generate random values in parallel
            Parallel.For(0, totalSize, i =>
            {
                double randomValue = RandomGen.Value.NextDouble();
                result.Data[i] = (randomValue * (maxValue - minValue)) + minValue;
            });

            return result;
        }

        /// <summary>
        /// Retrieves the specified row from a 2D tensor.
        /// </summary>
        /// <param name="rowIndex">The index of the row to retrieve.</param>
        /// <returns>A new tensor representing the row.</returns>
        /// <exception cref="InvalidOperationException">If the tensor is not 2D.</exception>
        /// <exception cref="ArgumentOutOfRangeException">If the row index is out of bounds.</exception>
        public Tensor GetRow(int rowIndex)
        {
            if (this.Shape.Length != 2)
            {
                throw new InvalidOperationException("GetRow can only be used on 2D tensors.");
            }

            int rows = this.Shape[0];
            int cols = this.Shape[1];

            if (rowIndex < 0 || rowIndex >= rows)
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of bounds.");
            }

            // Extract the row as a new 1D tensor
            double[] rowData = new double[cols];
            Array.Copy(this.Data, rowIndex * cols, rowData, 0, cols);

            return new Tensor(new int[] { 1, cols }, rowData);
        }

        /// <summary>
        /// Sets the specified row in a 2D tensor to the provided data.
        /// </summary>
        /// <param name="rowIndex">The index of the row to set.</param>
        /// <param name="rowData">The data to set for the row.</param>
        /// <exception cref="InvalidOperationException">If the tensor is not 2D.</exception>
        /// <exception cref="ArgumentOutOfRangeException">If the row index is out of bounds.</exception>
        /// <exception cref="ArgumentException">If the row data length does not match the tensor's column size.</exception>
        public void SetRow(int rowIndex, double[] rowData)
        {
            if (this.Shape.Length != 2)
            {
                throw new InvalidOperationException("SetRow can only be used on 2D tensors.");
            }

            int rows = this.Shape[0];
            int cols = this.Shape[1];

            if (rowIndex < 0 || rowIndex >= rows)
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of bounds.");
            }

            if (rowData.Length != cols)
            {
                throw new ArgumentException($"Row data length ({rowData.Length}) does not match the number of columns ({cols}).", nameof(rowData));
            }

            // Set the row data
            Array.Copy(rowData, 0, this.Data, rowIndex * cols, cols);
        }

        /// <summary>
        /// Generates a tensor containing a range of values.
        /// </summary>
        /// <param name="start">The starting value of the range.</param>
        /// <param name="end">The ending value of the range (exclusive).</param>
        /// <returns>A tensor containing the range of values.</returns>
        public Tensor Arange(int start, int end)
        {
            int length = end - start;
            double[] data = new double[length];

            Parallel.For(0, length, i =>
            {
                data[i] = start + i;
            });

            return new Tensor(new int[] { length }, data);
        }

        /// <summary>
        /// Excludes the values of the tensor that are in the specified range.
        /// If values within the range are closer to max, they become max, otherwise they become min.
        /// </summary>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The excluded tensor.</returns>
        public Tensor Exclude(double min, double max)
        {
            var excludedData = new double[this.Data.Length];

            var halfDistance = (max - min) / 2.0;

            Parallel.For(0, this.Data.Length, i =>
            {
                if (this.Data[i] >= min && this.Data[i] <= max)
                {
                    var distance = this.Data[i] - min;
                    if (distance > halfDistance)
                    {
                        excludedData[i] = max;
                    }
                    else
                    {
                        excludedData[i] = min;
                    }
                }
                else
                {
                    excludedData[i] = this.Data[i];
                }
            });

            return new Tensor(this.Shape, excludedData);
        }

        /// <summary>
        /// Clips the values of the tensor to be within the specified range.
        /// </summary>
        /// <param name="min">The minimum value to clip to.</param>
        /// <param name="max">The maximum value to clip to.</param>
        /// <returns>A new tensor with values clipped to the specified range.</returns>
        public Tensor Clip(double min, double max)
        {
            var clippedData = new double[this.Data.Length];
            var minArray = new double[this.Data.Length];
            var maxArray = new double[this.Data.Length];

            // Fill minArray and maxArray with appropriate values
            Array.Fill(minArray, min);
            Array.Fill(maxArray, max);

            // Perform element-wise min and max operations
            var clippedMin = new double[this.Data.Length];
            Vml.MinMag(this.Data.Length, this.Data, maxArray, clippedMin);
            Vml.MaxMag(this.Data.Length, clippedMin, minArray, clippedData);

            return new Tensor(this.Shape, clippedData);
        }

        /// <summary>
        /// Element-wise maximum between this tensor and a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to compare.</param>
        /// <returns>A new tensor containing the element-wise maximum values.</returns>
        public Tensor Max(double scalar)
        {
            // Allocate the result tensor
            var resultData = PradTools.AllocateArray(this.Data.Length);
            var resultTensor = new Tensor(this.Shape, resultData);

            // Create a scalar array for broadcasting the scalar value
            var scalarArray = PradTools.AllocateArray(this.Data.Length);
            Array.Fill(scalarArray, scalar);

            // Perform the element-wise maximum using MKLNET
            Vml.MaxMag(this.Data.Length, this.Data, scalarArray, resultData);

            return resultTensor;
        }

        /// <summary>
        /// Element-wise minimum between this tensor and a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to compare.</param>
        /// <returns>A new tensor containing the element-wise minimum values.</returns>
        public Tensor Min(double scalar)
        {
            // Allocate the result tensor
            var resultData = PradTools.AllocateArray(this.Data.Length);
            var resultTensor = new Tensor(this.Shape, resultData);

            // Create a scalar array for broadcasting the scalar value
            var scalarArray = PradTools.AllocateArray(this.Data.Length);
            Array.Fill(scalarArray, scalar);

            // Perform the element-wise minimum using MKLNET
            Vml.MinMag(this.Data.Length, this.Data, scalarArray, resultData);

            return resultTensor;
        }

        /// <summary>
        /// Sums the tensor elements along the specified axes.
        /// </summary>
        /// <param name="axes">The axes along which to sum the elements.</param>
        /// <returns>A new tensor with the summed elements.</returns>
        public Tensor Sum(int[] axes)
        {
            // Check if all axes are summed
            if (axes.Length == this.Shape.Length)
            {
                // Return the sum of all elements as a scalar (wrapped in a tensor)
                double totalSum = this.Data.Sum();
                return new Tensor(new int[] { 1 }, new double[] { totalSum });
            }

            // Step 1: Determine the shape of the resulting tensor after summing
            var newShape = this.Shape.ToList();

            if (axes.Length == 1 && axes[0] == -1)
            {
                axes[0] = newShape.Count - 1;
            }

            foreach (var axis in axes.OrderByDescending(a => a))
            {
                newShape.RemoveAt(axis);
            }

            if (newShape.Count == 0)
            {
                newShape.Add(1);
            }

            // Step 2: Create the result tensor with the new shape
            var resultData = new double[newShape.Aggregate(1, (a, b) => a * b)];
            var resultTensor = new Tensor(newShape.ToArray(), resultData);

            // Step 3: Calculate the strides for original tensor
            int[] strides = new int[this.Shape.Length];
            strides[strides.Length - 1] = 1;
            for (int i = strides.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * this.Shape[i + 1];
            }

            // Step 4: Sum the elements along the specified axes
            int[] currentIndices = new int[this.Shape.Length];
            this.SumRecursive(axes, 0, currentIndices, strides, resultData, 0);

            return resultTensor;
        }

        /// <summary>
        /// Performs matrix multiplication using MKL dgemm.
        /// </summary>
        /// <param name="other">The other tensor to multiply with.</param>
        /// <param name="alpha">The scalar multiplier for the product of A and B.</param>
        /// <param name="beta">The scalar multiplier for the existing values in the result tensor.</param>
        /// <returns>A new tensor resulting from the matrix multiplication.</returns>
        /// <exception cref="ArgumentException">If the tensors have incompatible shapes for multiplication.</exception>
        public Tensor MatrixMultiply(Tensor other, double alpha = 1.0, double beta = 0.0)
        {
            if (this.Shape.Length != 2 || other.Shape.Length != 2)
            {
                throw new ArgumentException("Matrix multiplication is only supported for 2D tensors.");
            }

            int m = this.Shape[0]; // Rows of A and C
            int k = this.Shape[1]; // Columns of A and rows of B
            int n = other.Shape[1]; // Columns of B and C

            if (k != other.Shape[0])
            {
                throw new ArgumentException("Incompatible shapes for matrix multiplication.");
            }

            var resultShape = new int[] { m, n };
            var resultData = new double[m * n];
            var result = new Tensor(resultShape, resultData);

            Blas.gemm(
                MKLNET.Layout.RowMajor,
                MKLNET.Trans.No,
                MKLNET.Trans.No,
                m,
                n,
                k,
                alpha,
                A: this.Data.AsSpan(),
                lda: k,
                B: other.Data.AsSpan(),
                ldb: n,
                beta,
                C: resultData.AsSpan(),
                ldc: n);

            return result;
        }

        /// <summary>
        /// Deep clone the tensor.
        /// </summary>
        /// <returns>The tensor.</returns>
        public Tensor DeepClone()
        {
            int[] newShape = (int[])this.Shape.Clone();
            double[] newData = (double[])this.Data.Clone();

            Array.Copy(this.Shape, newShape, this.Shape.Length);
            Buffer.BlockCopy(this.Data, 0, newData, 0, this.Data.Length * sizeof(double));

            return new Tensor(newShape, newData);
        }

        /// <summary>
        /// Replace the data with new data.
        /// </summary>
        /// <param name="newData">New data.</param>
        /// <exception cref="ArgumentException">Size doesn't match.</exception>
        public void ReplaceData(double[] newData)
        {
            if (this.GetTotalSize(this.Shape) != newData.Length)
            {
                throw new ArgumentException("Data length does not match shape size.");
            }

            this.Data = newData;
        }

        /// <summary>
        /// Converts to a Matrix.
        /// </summary>
        /// <returns>A matrix.</returns>
        public Matrix ToMatrix()
        {
            int totalSize = this.GetTotalSize(this.Shape);
            double[][] matrixData;

            if (this.Shape.Length == 1)
            {
                // 1D Tensor
                matrixData = new double[1][];
                matrixData[0] = new double[totalSize];
                Array.Copy(this.Data, matrixData[0], totalSize);
            }
            else
            {
                // ND Tensor
                int rows = 1;
                for (int i = 0; i < this.Shape.Length - 1; i++)
                {
                    rows *= this.Shape[i];
                }

                int cols = this.Shape[this.Shape.Length - 1];
                matrixData = new double[rows][];

                for (int i = 0; i < rows; i++)
                {
                    matrixData[i] = new double[cols];
                    Array.Copy(this.Data, i * cols, matrixData[i], 0, cols);
                }
            }

            return new Matrix(matrixData, this.Shape, this.Shape.Length);
        }

        /// <summary>
        /// Converts the Tensor to a DeepMatrix.
        /// Assumes the first dimension of the Tensor represents the depth of the DeepMatrix.
        /// </summary>
        /// <returns>The DeepMatrix.</returns>
        public DeepMatrix ToDeepMatrix()
        {
            if (this.Shape.Length != 3)
            {
                throw new InvalidOperationException("Tensor must have exactly 3 dimensions to convert to DeepMatrix (depth, rows, cols).");
            }

            int depth = this.Shape[0];
            int rows = this.Shape[1];
            int cols = this.Shape[2];

            Matrix[] matrices = new Matrix[depth];
            Parallel.For(0, depth, d =>
            {
                double[][] matrixData = new double[rows][];
                for (int r = 0; r < rows; r++)
                {
                    matrixData[r] = new double[cols];
                    for (int c = 0; c < cols; c++)
                    {
                        matrixData[r][c] = this[d, r, c];
                    }
                }

                matrices[d] = new Matrix(matrixData);
            });

            return new DeepMatrix(matrices);
        }

        /// <summary>
        /// Computes the mean of the tensor along the specified axis.
        /// </summary>
        /// <param name="axis">The specified axis.</param>
        /// <returns>A new tensor with the result.</returns>
        /// <exception cref="ArgumentException">Axis is out of bounds.</exception>
        public Tensor Mean(int axis)
        {
            if (axis < 0)
            {
                axis += this.Shape.Length;
            }

            if (axis < 0 || axis >= this.Shape.Length)
            {
                throw new ArgumentException("Axis is out of bounds for the tensor.");
            }

            int[] newShape = this.Shape.Where((_, idx) => idx != axis).ToArray();
            if (newShape.Length == 0)
            {
                newShape = new int[] { 1 };
            }

            int outerSize = this.Shape.Take(axis).Aggregate(1, (a, b) => a * b);
            int axisSize = this.Shape[axis];
            int innerSize = this.Shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b);

            double[] resultData = new double[outerSize * innerSize];

            double[] axisData = new double[innerSize];
            Array.Fill<double>(axisData, axisSize);

            Parallel.For(0, outerSize, outerIdx =>
            {
                double[] sum = new double[innerSize];
                int srcOffset = outerIdx * axisSize * innerSize;

                for (int axisIdx = 0; axisIdx < axisSize; axisIdx++)
                {
                    int offset = srcOffset + (axisIdx * innerSize);
                    Vml.Add(innerSize, sum, this.Data.AsSpan(offset, innerSize), sum);
                }

                int destOffset = outerIdx * innerSize;
                Vml.Div(innerSize, sum, axisData, resultData.AsSpan(destOffset, innerSize));
            });

            return new Tensor(newShape, resultData);
        }

        /// <summary>
        /// Sum the rows of the tensor.
        /// </summary>
        /// <returns>The tensor.</returns>
        public Tensor SumRows()
        {
            int[] shape = this.Shape;
            double[] resultData = new double[shape[0]];

            for (int i = 0; i < shape[0]; i++)
            {
                double sum = 0;
                for (int j = 0; j < shape[1]; j++)
                {
                    sum += this[i, j];
                }

                resultData[i] = sum;
            }

            return new Tensor(new int[] { shape[0], 1 }, resultData);
        }

        /// <summary>
        /// Samples from a multinomial distribution based on class probabilities with temperature scaling.
        /// </summary>
        /// <param name="numSamples">Number of samples to draw for each batch.</param>
        /// <param name="temperature">Temperature to control the sharpness of the distribution. Default is 1.0.</param>
        /// <returns>A tensor of shape [batch_size, num_samples] with the sampled class indices.</returns>
        /// <exception cref="ArgumentException">Thrown if the tensor is not 2D.</exception>
        public Tensor Multinomial(int numSamples, double temperature = 1.0)
        {
            if (this.Shape.Length != 2)
            {
                throw new ArgumentException("Tensor must be a 2D tensor with shape [batch_size, num_classes].");
            }

            int batchSize = this.Shape[0];
            int numClasses = this.Shape[1];

            var result = new Tensor(new int[] { batchSize, numSamples });

            // Scale logits by temperature (softening or sharpening the distribution)
            Tensor scaledLogits = temperature == 1.0 ? this : this.Divide(temperature);

            // Convert logits to probabilities using softmax
            var probabilities = scaledLogits.Softmax(axis: 1);

            Parallel.For(0, batchSize, batch =>
            {
                var cumulativeProbabilities = new double[numClasses];
                cumulativeProbabilities[0] = probabilities[batch, 0];

                // Compute cumulative distribution function (CDF)
                for (int i = 1; i < numClasses; i++)
                {
                    cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + probabilities[batch, i];
                }

                // Sample from the multinomial distribution
                for (int sample = 0; sample < numSamples; sample++)
                {
                    double u = RandomGen.Value.NextDouble();

                    for (int i = 0; i < numClasses; i++)
                    {
                        if (u <= cumulativeProbabilities[i])
                        {
                            result[batch, sample] = i;
                            break;
                        }
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Element-wise division of the tensor by a scalar (temperature scaling).
        /// </summary>
        /// <param name="scalar">The scalar value to divide the tensor by.</param>
        /// <returns>A new tensor with each element divided by the scalar.</returns>
        public Tensor Divide(double scalar)
        {
            var result = new Tensor(this.Shape);
            Parallel.For(0, this.Data.Length, i =>
            {
                result.Data[i] = this.Data[i] / scalar;
            });
            return result;
        }

        /// <summary>
        /// Unstacks the tensor along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to unstack.</param>
        /// <returns>An array of tensors resulting from the unstack operation.</returns>
        /// <exception cref="ArgumentException">If the axis is out of bounds.</exception>
        public Tensor[] Unstack(int axis = 0)
        {
            int[] shape = this.Shape;
            int rank = shape.Length;

            // Handle negative axis
            if (axis < 0)
            {
                axis = rank + axis;
            }

            // Validate axis
            if (axis < 0 || axis >= rank)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Determine the output shape for each unstacked tensor
            int[] outputShape = new int[rank - 1];
            for (int i = 0, j = 0; i < rank; i++)
            {
                if (i != axis)
                {
                    outputShape[j++] = shape[i];
                }
            }

            int numTensors = shape[axis];
            Tensor[] result = new Tensor[numTensors];

            // Calculate the strides for the tensor
            int[] strides = new int[rank];
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            Parallel.For(0, numTensors, i =>
            {
                var dataSize = this.Data.Length / numTensors;
                double[] data = new double[dataSize];
                var resultTensor = new Tensor(outputShape, data);

                int[] sourceIndices = new int[rank];
                int[] resultIndices = new int[outputShape.Length];

                int sourceIndex = i * strides[axis];
                for (int j = 0; j < dataSize; j++)
                {
                    int temp = j;
                    for (int k = outputShape.Length - 1; k >= 0; k--)
                    {
                        resultIndices[k] = temp % outputShape[k];
                        temp /= outputShape[k];
                    }

                    for (int k = 0, l = 0; k < rank; k++)
                    {
                        if (k == axis)
                        {
                            sourceIndices[k] = i;
                        }
                        else
                        {
                            sourceIndices[k] = resultIndices[l++];
                        }
                    }

                    resultTensor.Data[j] = this.Data[this.GetIndex(sourceIndices)];
                }

                result[i] = resultTensor;
            });

            return result;
        }

        private void SumRecursive(int[] axes, int depth, int[] currentIndices, int[] strides, double[] resultData, int resultIndex)
        {
            if (depth == this.Shape.Length)
            {
                int dataIndex = 0;
                for (int i = 0; i < currentIndices.Length; i++)
                {
                    dataIndex += currentIndices[i] * strides[i];
                }

                resultData[resultIndex] += this.Data[dataIndex];
                return;
            }

            if (axes.Contains(depth))
            {
                for (int i = 0; i < this.Shape[depth]; i++)
                {
                    currentIndices[depth] = i;
                    this.SumRecursive(axes, depth + 1, currentIndices, strides, resultData, resultIndex);
                }
            }
            else
            {
                int stride = (depth == this.Shape.Length - 1) ? 1 :
                    Enumerable.Range(depth + 1, this.Shape.Length - depth - 1)
                              .Where(i => !axes.Contains(i))
                              .Aggregate(1, (acc, i) => acc * this.Shape[i]);

                for (int i = 0; i < this.Shape[depth]; i++)
                {
                    currentIndices[depth] = i;
                    this.SumRecursive(axes, depth + 1, currentIndices, strides, resultData, resultIndex);
                    resultIndex += stride;
                }
            }
        }
    }
}

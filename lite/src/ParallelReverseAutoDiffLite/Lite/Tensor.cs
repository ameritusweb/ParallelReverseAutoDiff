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
            this.Data = new float[this.GetTotalSize(shape)];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class.
        /// </summary>
        /// <param name="original">The original tensor.</param>
        public Tensor(Tensor original)
        {
            this.Shape = new int[original.Shape.Length];
            Array.Copy(original.Shape, this.Shape, original.Shape.Length);
            this.Data = new float[original.Data.Length];
            Buffer.BlockCopy(original.Data, 0, this.Data, 0, original.Data.Length * sizeof(float));
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor"/> class.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="data">The data.</param>
        /// <exception cref="ArgumentException">The data length does not match.</exception>
        public Tensor(int[] shape, float[] data)
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
        public Tensor(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);

            this.Shape = new int[] { rows, cols };
            this.Data = new float[rows * cols];

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
        public Tensor(int[] shape, float value)
        {
            this.Shape = shape;
            int totalSize = shape.Aggregate(1, (a, b) => a * b);
            this.Data = new float[totalSize];
            Array.Fill(this.Data, value);
        }

        /// <summary>
        /// Gets the data for the tensor.
        /// </summary>
        public float[] Data { get; private set; }

        /// <summary>
        /// An indexer.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>The data value.</returns>
        public float this[params int[] indices]
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
        public static Tensor operator >(Tensor a, float scalar)
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
        public static Tensor operator <(Tensor a, float scalar)
        {
            var result = new Tensor(a.Shape);
            Parallel.For(0, a.Data.Length, i =>
            {
                result.Data[i] = a.Data[i] < scalar ? 1 : 0;
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
            var outputData = new float[totalSize];
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
                Buffer.BlockCopy(tensorData, 0, outputData, offset * sizeof(float), tensorData.Length * sizeof(float));
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
        /// Concatenates a list of tensors along a specified axis.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>A tensor.</returns>
        /// <exception cref="ArgumentException">Must contain at least two tensors.</exception>
        public static Tensor Concat(Tensor[] tensors, int axis = 0)
        {
            if (tensors == null || tensors.Length < 2)
            {
                throw new ArgumentException("The input list of tensors must contain at least two tensors.");
            }

            var shape = tensors[0].Shape;
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

            // Check compatibility and calculate the total size along the concatenation axis
            int concatDimSize = 0;
            foreach (var tensor in tensors)
            {
                if (tensor.Shape.Length != rank)
                {
                    throw new ArgumentException("All input tensors must have the same rank.");
                }

                for (int i = 0; i < rank; i++)
                {
                    if (i != axis && tensor.Shape[i] != shape[i])
                    {
                        throw new ArgumentException("All input tensors must have the same shape except along the concatenation axis.");
                    }
                }

                concatDimSize += tensor.Shape[axis];
            }

            // Determine the output shape
            var outputShape = new int[rank];
            Array.Copy(shape, outputShape, rank);
            outputShape[axis] = concatDimSize;

            var outputData = new float[outputShape.Aggregate((a, b) => a * b)];
            var outputTensor = new Tensor(outputShape, outputData);

            // Calculate the size of each slice along the concatenation axis
            int sliceSize = 1;
            for (int i = axis + 1; i < rank; i++)
            {
                sliceSize *= outputShape[i];
            }

            // Calculate the number of slices
            int numSlices = 1;
            for (int i = 0; i < axis; i++)
            {
                numSlices *= outputShape[i];
            }

            // Concatenate the data
            int outputOffset = 0;
            for (int slice = 0; slice < numSlices; slice++)
            {
                foreach (var tensor in tensors)
                {
                    int tensorSliceSize = sliceSize * tensor.Shape[axis];
                    int inputOffset = slice * tensorSliceSize;
                    Buffer.BlockCopy(tensor.Data, inputOffset * sizeof(float), outputData, outputOffset * sizeof(float), tensorSliceSize * sizeof(float));
                    outputOffset += tensorSliceSize;
                }
            }

            return outputTensor;
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
            float[] flatArray = new float[tensors.Length * indices.Length * shape[0]];
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
        public static Tensor CreateMask(Tensor tensor, Func<float, bool> conditionFunc)
        {
            var maskData = new float[tensor.Data.Length];
            Parallel.For(0, tensor.Data.Length, i =>
            {
                maskData[i] = conditionFunc(tensor.Data[i]) ? 1.0f : 0.0f;
            });

            return new Tensor(tensor.Shape, maskData);
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
            float[] data = new float[length];

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
            var excludedData = new float[this.Data.Length];

            var halfDistance = (max - min) / 2.0;

            Parallel.For(0, this.Data.Length, i =>
            {
                if (this.Data[i] >= min && this.Data[i] <= max)
                {
                    var distance = this.Data[i] - min;
                    if (distance > halfDistance)
                    {
                        excludedData[i] = (float)max;
                    }
                    else
                    {
                        excludedData[i] = (float)min;
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
            var clippedData = new float[this.Data.Length];
            var minArray = new float[this.Data.Length];
            var maxArray = new float[this.Data.Length];

            // Fill minArray and maxArray with appropriate values
            Array.Fill(minArray, (float)min);
            Array.Fill(maxArray, (float)max);

            // Perform element-wise min and max operations
            var clippedMin = new float[this.Data.Length];
            Vml.MinMag(this.Data.Length, this.Data, maxArray, clippedMin);
            Vml.MaxMag(this.Data.Length, clippedMin, minArray, clippedData);

            return new Tensor(this.Shape, clippedData);
        }

        /// <summary>
        /// Broadcasts the tensor to a specified shape.
        /// </summary>
        /// <param name="newShape">The new shape to broadcast to.</param>
        /// <returns>A new tensor broadcasted to the specified shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the new shape is not compatible for broadcasting.</exception>
        public Tensor BroadcastTo(int[] newShape)
        {
            if (newShape.Length < this.Shape.Length)
            {
                throw new ArgumentException("New shape must be of the same rank or higher than the current shape.");
            }

            // Ensure the shape is compatible for broadcasting
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (this.Shape[i] != 1 && this.Shape[i] != newShape[newShape.Length - this.Shape.Length + i])
                {
                    throw new ArgumentException("Shapes are not compatible for broadcasting.");
                }
            }

            // Create the new data array
            int newTotalSize = newShape.Aggregate(1, (a, b) => a * b);
            var broadcastedData = new float[newTotalSize];

            // Fill the new data array by repeating the elements
            int[] indices = new int[newShape.Length];
            Parallel.For(0, newTotalSize, i =>
            {
                int oldIndex = 0;
                int stride = 1;
                for (int j = this.Shape.Length - 1; j >= 0; j--)
                {
                    int dim = this.Shape.Length - 1 - j;
                    if (this.Shape[dim] != 1)
                    {
                        oldIndex += (indices[newShape.Length - 1 - j] % this.Shape[dim]) * stride;
                    }

                    stride *= this.Shape[dim];
                }

                broadcastedData[i] = this.Data[oldIndex];

                // Increment the indices
                for (int k = newShape.Length - 1; k >= 0; k--)
                {
                    indices[k]++;
                    if (indices[k] < newShape[k])
                    {
                        break;
                    }

                    indices[k] = 0;
                }
            });

            return new Tensor(newShape, broadcastedData);
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
                float totalSum = this.Data.Sum();
                return new Tensor(new int[] { 1 }, new float[] { totalSum });
            }

            // Step 1: Determine the shape of the resulting tensor after summing
            var newShape = this.Shape.ToList();
            foreach (var axis in axes.OrderByDescending(a => a))
            {
                newShape.RemoveAt(axis);
            }

            if (newShape.Count == 0)
            {
                newShape.Add(1);
            }

            // Step 2: Create the result tensor with the new shape
            var resultData = new float[newShape.Aggregate(1, (a, b) => a * b)];
            var resultTensor = new Tensor(newShape.ToArray(), resultData);

            // Step 3: Calculate the strides for original tensor
            var strides = new int[this.Shape.Length];
            strides[this.Shape.Length - 1] = 1;
            for (int i = this.Shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * this.Shape[i + 1];
            }

            // Step 4: Sum the elements along the specified axes
            Parallel.For(0, resultTensor.Data.Length, i =>
            {
                var resultIndex = new int[resultTensor.Shape.Length];
                var inputIndex = new int[this.Shape.Length];
                int remainingIndex = i;
                for (int j = resultTensor.Shape.Length - 1; j >= 0; j--)
                {
                    resultIndex[j] = remainingIndex % resultTensor.Shape[j];
                    remainingIndex /= resultTensor.Shape[j];
                }

                for (int j = 0, k = 0; j < this.Shape.Length; j++)
                {
                    if (axes.Contains(j))
                    {
                        inputIndex[j] = 0;
                    }
                    else
                    {
                        inputIndex[j] = resultIndex[k++];
                    }
                }

                var sum = 0.0f;
                this.SumRecursive(inputIndex, resultIndex, 0, ref sum, strides);
                resultTensor.Data[i] = sum;
            });

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
        public Tensor MatrixMultiply(Tensor other, float alpha = 1.0f, float beta = 0.0f)
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
            var resultData = new float[m * n];
            var result = new Tensor(resultShape, resultData);

            Blas.gemm(
                Layout.RowMajor,
                Trans.No,
                Trans.No,
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
            float[] newData = (float[])this.Data.Clone();

            Array.Copy(this.Shape, newShape, this.Shape.Length);
            Buffer.BlockCopy(this.Data, 0, newData, 0, this.Data.Length * sizeof(float));

            return new Tensor(newShape, newData);
        }

        /// <summary>
        /// Replace the data with new data.
        /// </summary>
        /// <param name="newData">New data.</param>
        /// <exception cref="ArgumentException">Size doesn't match.</exception>
        public void ReplaceData(float[] newData)
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
            float[][] matrixData;

            if (this.Shape.Length == 1)
            {
                // 1D Tensor
                matrixData = new float[1][];
                matrixData[0] = new float[totalSize];
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
                matrixData = new float[rows][];

                for (int i = 0; i < rows; i++)
                {
                    matrixData[i] = new float[cols];
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
                float[][] matrixData = new float[rows][];
                for (int r = 0; r < rows; r++)
                {
                    matrixData[r] = new float[cols];
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

            float[] resultData = new float[outerSize * innerSize];

            float[] axisData = new float[innerSize];
            Array.Fill<float>(axisData, axis);

            Parallel.For(0, outerSize, outerIdx =>
            {
                float[] sum = new float[innerSize];
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
            float[] resultData = new float[shape[0]];

            for (int i = 0; i < shape[0]; i++)
            {
                float sum = 0;
                for (int j = 0; j < shape[1]; j++)
                {
                    sum += this[i, j];
                }

                resultData[i] = sum;
            }

            return new Tensor(new int[] { shape[0], 1 }, resultData);
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
                float[] data = new float[dataSize];
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

        private void SumRecursive(int[] inputIndex, int[] resultIndex, int currentAxis, ref float sum, int[] strides)
        {
            if (currentAxis == this.Shape.Length)
            {
                int flatIndex = 0;
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    flatIndex += inputIndex[i] * strides[i];
                }

                sum += this.Data[flatIndex];
            }
            else
            {
                if (resultIndex.Contains(currentAxis))
                {
                    for (int i = 0; i < this.Shape[currentAxis]; i++)
                    {
                        inputIndex[currentAxis] = i;
                        this.SumRecursive(inputIndex, resultIndex, currentAxis + 1, ref sum, strides);
                    }
                }
                else
                {
                    this.SumRecursive(inputIndex, resultIndex, currentAxis + 1, ref sum, strides);
                }
            }
        }
    }
}

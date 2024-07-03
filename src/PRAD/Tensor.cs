//------------------------------------------------------------------------------
// <copyright file="Tensor.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using MKLNET;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A flat tensor.
    /// </summary>
    public class Tensor
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
        /// Gets the shape of the tensor.
        /// </summary>
        public int[] Shape { get; private set; }

        /// <summary>
        /// Gets the debug info.
        /// </summary>
        public string DebugInfo { get; private set; }

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
        /// Element-wise multiply two tensors together.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns>The resultant tensor.</returns>
        public static Tensor operator *(Tensor a, Tensor b)
        {
            return a.ElementwiseMultiply(b);
        }

        /// <summary>
        /// Slice 3-D tensors to form tensors of the specified slice sizes.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="sliceSizes">The slice sizes.</param>
        /// <returns>The tensor list.</returns>
        /// <exception cref="ArgumentException">Slice sizes must be a 3-tuple.</exception>
        public static List<Tensor> Slice3DTensors(Tensor[] tensors, int[] sliceSizes)
        {
            if (sliceSizes.Length != 3)
            {
                throw new ArgumentException("Slice sizes must be a 3-tuple.");
            }

            var slices = new List<Tensor>();

            foreach (var tensor in tensors)
            {
                if (tensor.Shape.Length != 3)
                {
                    throw new ArgumentException("All input tensors must be 3-dimensional.");
                }

                // Calculate the number of slices for each dimension
                int[] numSlices = tensor.CalculateNumberOfSlices(sliceSizes);

                // Generate all possible slice start indices
                List<int[]> indicesList = tensor.GenerateSliceStartIndices(numSlices);

                // Extract slices
                foreach (var start in indicesList)
                {
                    for (int i = 0; i < start.Length; i++)
                    {
                        start[i] *= sliceSizes[i];
                    }

                    var sliceTensor = tensor.Slice(start, sliceSizes);
                    slices.Add(sliceTensor);
                }
            }

            return slices;
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

            var outputData = new double[outputShape.Aggregate((a, b) => a * b)];
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
                    Buffer.BlockCopy(tensor.Data, inputOffset * sizeof(double), outputData, outputOffset * sizeof(double), tensorSliceSize * sizeof(double));
                    outputOffset += tensorSliceSize;
                }
            }

            return outputTensor;
        }

        /// <summary>
        /// Concatenates slices of the input tensors along the specified axis.
        /// </summary>
        /// <param name="tensors">The array of tensors to be concatenated.</param>
        /// <param name="axisRange">The range of axes to be considered for slicing, formatted as "start:end".</param>
        /// <param name="concatAxis">The axis along which the concatenation is performed.</param>
        /// <returns>A new tensor that is the result of concatenating the input tensors along the specified axis.</returns>
        /// <exception cref="ArgumentException">Thrown when no tensors are provided or when invalid axis range or concat axis are specified.</exception>
        public static Tensor ConcatSlices(Tensor[] tensors, string axisRange, int concatAxis)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException("At least one tensor must be provided.");
            }

            (int start, int end) = ParseAxisRange(axisRange);
            int maxRank = tensors.Max(t => t.Shape.Length);

            // Adjust negative indices
            start = start < 0 ? maxRank + start : start;
            end = end <= 0 ? maxRank + end : end;

            // Ensure end is not less than start
            end = Math.Max(end, start);

            if (start < 0 || start >= maxRank || end > maxRank || start >= end || concatAxis < 0 || concatAxis >= maxRank)
            {
                throw new ArgumentException("Invalid axis range or concat axis.");
            }

            int scenario = DetermineScenario(start, end, concatAxis, maxRank);

            return scenario switch
            {
                1 => StandardConcat(tensors, concatAxis),
                2 => ConcatByUnstackAndExpand(tensors, 0, 0),
                3 => ConcatByUnstackAndExpand(tensors, 0, 1),
                4 => ConcatByUnstackAndExpand(tensors, 0, 1, true),
                5 => ConcatByExpansion(tensors, 0),
                6 => ConcatByUnstackAndExpand(tensors, 0, -2, false, true),
                7 => ConcatByUnstackAndExpand(tensors, 0, -1, true),
                _ => throw new NotImplementedException($"Scenario {scenario} not implemented."),
            };
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
            var broadcastedData = new double[newTotalSize];

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
            var resultData = new double[newShape.Aggregate(1, (a, b) => a * b)];
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

                var sum = 0.0;
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
        /// Computes the element-wise reciprocal of the tensor.
        /// </summary>
        /// <returns>The resultant tensor.</returns>
        public Tensor Reciprocal()
        {
            Tensor result = new Tensor(this.Shape);

            MKLNET.Vml.Inv(this.Data.Length, this.Data, result.Data);

            return result;
        }

        /// <summary>
        /// Expands the dimensions.
        /// </summary>
        /// <param name="axis">The axis.</param>
        /// <returns>The resultant tensor.</returns>
        public Tensor ExpandDims(int axis = -1)
        {
            // Handle negative axis
            if (axis < 0)
            {
                axis = this.Shape.Length + 1 + axis;
            }

            // Validate axis
            if (axis < 0 || axis > this.Shape.Length)
            {
                throw new ArgumentException($"Invalid axis {axis} for tensor with {this.Shape.Length} dimensions.");
            }

            // Create new shape
            int[] newShape = new int[this.Shape.Length + 1];
            for (int i = 0; i < axis; i++)
            {
                newShape[i] = this.Shape[i];
            }

            newShape[axis] = 1;
            for (int i = axis; i < this.Shape.Length; i++)
            {
                newShape[i + 1] = this.Shape[i];
            }

            // Create new tensor with expanded dimensions
            Tensor result = new Tensor(newShape, this.Data);

            return result;
        }

        /// <summary>
        /// Computes the elementwise absolute value of the tensor.
        /// </summary>
        /// <returns>The resultant tensor.</returns>
        public Tensor Abs()
        {
            var result = new Tensor(this.Shape);
            Vml.Abs(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the elementwise sign of the tensor.
        /// </summary>
        /// <returns>The resultant tensor.</returns>
        public Tensor Sign()
        {
            var result = new Tensor(this.Shape);
            Parallel.For(0, this.Data.Length, i =>
            {
                result.Data[i] = Math.Sign(this.Data[i]);
            });

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
        /// Computes the element-wise exponential of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor Exp()
        {
            var result = new Tensor(this.Shape);
            Vml.Exp(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise natural logarithm of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor Ln()
        {
            var result = new Tensor(this.Shape);
            Vml.Ln(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise logarithm base 10 of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor Log()
        {
            var result = new Tensor(this.Shape);
            Vml.Log10(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Removes dimensions of size 1 from the tensor.
        /// </summary>
        /// <param name="axes">The axes to squeeze. If null, all axes of size 1 will be removed.</param>
        /// <returns>A new tensor with the specified dimensions removed.</returns>
        public Tensor Squeeze(int[]? axes = null)
        {
            if (axes == null)
            {
                axes = Enumerable.Range(0, this.Shape.Length).Where(i => this.Shape[i] == 1).ToArray();
            }

            var newShape = this.Shape.Where((dim, index) => !axes.Contains(index)).ToArray();
            return this.Reshape(newShape);
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
        /// Gathers slices from the tensor along the specified axis.
        /// </summary>
        /// <param name="indices">The indices of elements to gather.</param>
        /// <param name="axis">The axis along which to gather slices.</param>
        /// <returns>A new tensor with the gathered slices.</returns>
        public Tensor Gather(Tensor indices, int axis = 0)
        {
            var debugInfo = new StringBuilder();

            debugInfo.AppendLine($"Starting Gather operation on axis {axis}");
            debugInfo.AppendLine($"Input tensor shape: [{string.Join(", ", this.Shape)}]");
            debugInfo.AppendLine($"Input tensor data: [{string.Join(", ", this.Data)}]");
            debugInfo.AppendLine($"Indices shape: [{string.Join(", ", indices.Shape)}]");
            debugInfo.AppendLine($"Indices data: [{string.Join(", ", indices.Data)}]");

            if (axis < 0 || axis >= this.Shape.Length)
            {
                throw new ArgumentException($"Axis value {axis} is out of bounds for tensor with {this.Shape.Length} dimensions.");
            }

            // Handle negative indices and validate
            var indicesData = indices.Data.Select(i => i < 0 ? this.Shape[axis] + (int)i : (int)i).ToArray();
            debugInfo.AppendLine($"Processed indices: [{string.Join(", ", indicesData)}]");

            foreach (var index in indicesData)
            {
                if (index < 0 || index >= this.Shape[axis])
                {
                    throw new ArgumentException($"Index {index} is out of bounds for axis {axis} with size {this.Shape[axis]}.");
                }
            }

            // Calculate the result shape
            var resultShape = new int[this.Shape.Length + indices.Shape.Length - 1];
            Array.Copy(this.Shape, 0, resultShape, 0, axis);
            Array.Copy(indices.Shape, 0, resultShape, axis, indices.Shape.Length);
            Array.Copy(this.Shape, axis + 1, resultShape, axis + indices.Shape.Length, this.Shape.Length - axis - 1);
            debugInfo.AppendLine($"Calculated result shape: [{string.Join(", ", resultShape)}]");

            var result = new Tensor(resultShape);

            // Calculate strides for the original tensor
            int[] strides = new int[this.Shape.Length];
            strides[this.Shape.Length - 1] = 1;
            for (int i = this.Shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * this.Shape[i + 1];
            }

            debugInfo.AppendLine($"Calculated strides: [{string.Join(", ", strides)}]");

            int ComputeOutputIndex(int[] indices, int[] shape)
            {
                int index = 0;
                int stride = 1;
                for (int i = shape.Length - 1; i >= 0; i--)
                {
                    index += indices[i] * stride;
                    stride *= shape[i];
                }

                return index;
            }

            // Gather data
            ParallelOptions options = new ParallelOptions() { MaxDegreeOfParallelism = 1 };
            Parallel.For(0, result.Data.Length, options, i =>
            {
                // Initialize an array to hold the multi-dimensional indices for the current result element
                int[] resultIndices = new int[resultShape.Length];

                // Temporary variable to hold the current flat index
                int temp = i;

                // Convert the flat index 'i' to multi-dimensional indices
                for (int j = resultShape.Length - 1; j >= 0; j--)
                {
                    // Calculate the index for this dimension
                    resultIndices[j] = temp % resultShape[j];

                    // Update temp for the next iteration
                    temp /= resultShape[j];
                }

                // Initialize an array to hold the corresponding input tensor indices
                int[] inputIndices = new int[this.Shape.Length];

                // Variable to keep track of which gather index we're currently using
                // Calculate the gather dimension index outside the loop
                int gatherDimIndex = resultIndices[axis] % indices.Shape[0];

                // Calculate the input tensor indices based on the result indices
                for (int j = 0; j < this.Shape.Length; j++)
                {
                    if (j < axis)
                    {
                        // For dimensions before the gather axis, use the result index directly
                        inputIndices[j] = resultIndices[j];
                    }
                    else if (j == axis)
                    {
                        // For the gather axis, use the appropriate index from indicesData
                        inputIndices[j] = indicesData[gatherDimIndex];

                        // Move to the next gather index, wrapping around if necessary
                        // gatherDimIndex = (gatherDimIndex + 1) % indices.Shape[0];
                        inputIndices[j] = indicesData[gatherDimIndex];
                    }
                    else
                    {
                        // For dimensions after the gather axis, offset by the number of gathered dimensions
                        // inputIndices[j] = resultIndices[j + indices.Shape.Length - 1];
                        inputIndices[j] = resultIndices[j];
                    }
                }

                // Calculate the flat index for the input tensor
                int inputIndex = ComputeOutputIndex(inputIndices, this.Shape);

                // Calculate the flat index for the result tensor
                int outputIndex = ComputeOutputIndex(resultIndices, resultShape);

                // Copy the data from the input tensor to the result tensor
                result.Data[outputIndex] = this.Data[inputIndex];
            });

            debugInfo.AppendLine("Gather operation completed");
            debugInfo.AppendLine($"Result tensor shape: [{string.Join(", ", result.Shape)}]");
            debugInfo.AppendLine($"Result tensor data: [{string.Join(", ", result.Data)}]");

            // Store the debug info in the result tensor
            result.DebugInfo = debugInfo.ToString();

            return result;
        }

        /// <summary>
        /// Tiles the tensor along each dimension as specified by multiples.
        /// </summary>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>A new tensor that is tiled according to multiples.</returns>
        /// <exception cref="ArgumentException">Invalid multiples or shape mismatch.</exception>
        public Tensor Tile(int[] multiples)
        {
            if (multiples.Length != this.Shape.Length)
            {
                throw new ArgumentException("Length of multiples must match the number of dimensions of the tensor.");
            }

            foreach (var multiple in multiples)
            {
                if (multiple <= 0)
                {
                    throw new ArgumentException("All multiples must be positive integers.");
                }
            }

            int[] newShape = new int[this.Shape.Length];
            for (int i = 0; i < this.Shape.Length; i++)
            {
                newShape[i] = this.Shape[i] * multiples[i];
            }

            var result = new Tensor(newShape);
            int[] indices = new int[this.Shape.Length];
            int[] resultIndices = new int[this.Shape.Length];

            void TileRecursive(int dim)
            {
                if (dim == this.Shape.Length)
                {
                    double value = this[indices];
                    result[resultIndices] = value;
                }
                else
                {
                    for (int i = 0; i < this.Shape[dim]; i++)
                    {
                        indices[dim] = i;
                        for (int j = 0; j < multiples[dim]; j++)
                        {
                            resultIndices[dim] = i + (j * this.Shape[dim]);
                            TileRecursive(dim + 1);
                        }
                    }
                }
            }

            TileRecursive(0);
            return result;
        }

        /// <summary>
        /// Computes the element-wise addition of two tensors using MKL.NET.
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseAdd(Tensor other)
        {
            this.CheckShapeCompatibility(other);
            var result = new Tensor(this.Shape);
            Vml.Add(this.Data.Length, this.Data, other.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise subtraction of two tensors using MKL.NET.
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseSub(Tensor other)
        {
            this.CheckShapeCompatibility(other);
            var result = new Tensor(this.Shape);
            Vml.Sub(this.Data.Length, this.Data, other.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise multiplication of two tensors using MKL.NET.
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseMultiply(Tensor other)
        {
            this.CheckShapeCompatibility(other);
            var result = new Tensor(this.Shape);
            Vml.Mul(this.Data.Length, this.Data, other.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise division of two tensors using MKL.NET.
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseDivide(Tensor other)
        {
            this.CheckShapeCompatibility(other);
            var result = new Tensor(this.Shape);
            Vml.Div(this.Data.Length, this.Data, other.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise square of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseSquare()
        {
            var result = new Tensor(this.Shape);
            Vml.Mul(this.Data.Length, this.Data, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise square root of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseSquareRoot()
        {
            var result = new Tensor(this.Shape);
            Vml.Sqrt(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise sin of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseSin()
        {
            var result = new Tensor(this.Shape);
            Vml.Sin(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise cos of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseCos()
        {
            var result = new Tensor(this.Shape);
            Vml.Cos(this.Data.Length, this.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Computes the element-wise atan2 of the tensor using MKL.NET.
        /// </summary>
        /// <param name="x">The other tensor.</param>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseAtan2(Tensor x)
        {
            var result = new Tensor(this.Shape);
            Vml.Atan2(this.Data.Length, this.Data, x.Data, result.Data);
            return result;
        }

        /// <summary>
        /// Prints the tensor as C# code.
        /// </summary>
        /// <param name="decimals">The number of decimal places to round to.</param>
        /// <returns>The code.</returns>
        public string PrintCode(int decimals = 4)
        {
            var shapeStr = string.Join(", ", this.Shape);
            var dataStr = string.Join(", ", this.Data.Select(d => Math.Round(d, decimals).ToString().PadRight(decimals + 6, ' ')));

            return $"new Tensor(new int[] {{ {shapeStr} }}, new double[] {{ {dataStr} }})";
        }

        /// <summary>
        /// Prints the tensor to the console.
        /// </summary>
        public void Print()
        {
            int totalSize = this.GetTotalSize(this.Shape);
            for (int i = 0; i < totalSize; i++)
            {
                if (i % this.Shape[this.Shape.Length - 1] == 0)
                {
                    Console.WriteLine();
                }

                Console.Write($"{this.Data[i]} ");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Gathers slices from the tensor along the specified indices.
        /// </summary>
        /// <param name="indices">The tensor containing the indices.</param>
        /// <returns>A new tensor with the gathered slices.</returns>
        public Tensor GatherNd(Tensor indices)
        {
            if (indices.Shape[^1] != this.Shape.Length)
            {
                throw new ArgumentException("The last dimension of indices must match the rank of params.");
            }

            int[] resultShape = indices.Shape.Take(indices.Shape.Length - 1).ToArray();
            int resultSize = resultShape.Aggregate(1, (a, b) => a * b);
            var result = new Tensor(resultShape);

            Parallel.For(0, resultSize, i =>
            {
                int[] index = this.GetMultiDimensionalIndices(i, resultShape);
                int[] sourceIndex = new int[indices.Shape[^1]];

                for (int j = 0; j < sourceIndex.Length; j++)
                {
                    sourceIndex[j] = (int)indices[index.Concat(new int[] { j }).ToArray()];
                }

                result[index] = this[sourceIndex];
            });

            return result;
        }

        /// <summary>
        /// Reshapes the tensor to the new shape.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        /// <returns>A new tensor with the reshaped data.</returns>
        public Tensor Reshape(int[] newShape)
        {
            int newTotalSize = this.GetTotalSize(newShape);
            int currentTotalSize = this.Data.Length;

            if (newTotalSize != currentTotalSize && !newShape.Contains(-1))
            {
                throw new ArgumentException("New shape is incompatible with the total number of elements.");
            }

            if (newShape.Count(x => x == -1) > 1)
            {
                throw new ArgumentException("Only one dimension can be inferred.");
            }

            if (newShape.Contains(-1))
            {
                int inferredDimIndex = Array.IndexOf(newShape, -1);
                int inferredDim = currentTotalSize / this.GetTotalSize(newShape.Where((val, idx) => idx != inferredDimIndex).ToArray());
                newShape[inferredDimIndex] = inferredDim;
            }

            var result = new Tensor(newShape, this.Data);
            return result;
        }

        /// <summary>
        /// Extracts a slice from the tensor based on start indices, slice sizes, and optional strides.
        /// </summary>
        /// <param name="begin">The starting indices for each axis.</param>
        /// <param name="size">The lengths of the slice along each axis.</param>
        /// <param name="strides">The step size for each axis (default is 1).</param>
        /// <returns>A new tensor that is a slice of the original tensor.</returns>
        public Tensor Slice(int[] begin, int[] size, int[]? strides = null)
        {
            if (begin.Length != this.Shape.Length || size.Length != this.Shape.Length || (strides != null && strides.Length != this.Shape.Length))
            {
                throw new ArgumentException("The lengths of begin, size, and strides must match the number of dimensions of the tensor.");
            }

            strides = strides ?? Enumerable.Repeat(1, this.Shape.Length).ToArray();
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (strides[i] == 0)
                {
                    throw new ArgumentException("Stride cannot be zero.");
                }
            }

            int[] resultShape = new int[this.Shape.Length];
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (begin[i] < 0)
                {
                    begin[i] += this.Shape[i];
                }

                if (size[i] < 0)
                {
                    size[i] = (this.Shape[i] - begin[i]) / Math.Abs(strides[i]);
                }

                if (begin[i] < 0 || begin[i] >= this.Shape[i] ||
                    (begin[i] + ((size[i] - 1) * strides[i]) < 0 || begin[i] + ((size[i] - 1) * strides[i]) >= this.Shape[i]))
                {
                    throw new ArgumentException("The slice extends beyond the boundaries of the tensor.");
                }

                resultShape[i] = size[i];
            }

            var result = new Tensor(resultShape);

            Parallel.For(0, this.GetTotalSize(resultShape), resultIndex =>
            {
                int[] resultIndices = this.GetMultiDimensionalIndices(resultIndex, resultShape);
                int[] sourceIndices = new int[resultIndices.Length];
                for (int i = 0; i < resultIndices.Length; i++)
                {
                    sourceIndices[i] = begin[i] + (resultIndices[i] * strides[i]);
                }

                result[resultIndices] = this[sourceIndices];
            });

            return result;
        }

        /// <summary>
        /// Splits the tensor into multiple tensors along the specified axis.
        /// </summary>
        /// <param name="groupSize">The group size.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The tensors.</returns>
        public Tensor[] Split(int groupSize, int axis = 0)
        {
            if (axis < 0 || axis >= this.Shape.Length)
            {
                throw new ArgumentException("Axis is out of bounds for the tensor.");
            }

            if (this.Shape[axis] % groupSize != 0)
            {
                throw new ArgumentException("The specified axis dimension must be divisible by the group size.");
            }

            int numGroups = this.Shape[axis] / groupSize;
            Tensor[] result = new Tensor[numGroups];

            int[] begin = new int[this.Shape.Length];
            int[] size = (int[])this.Shape.Clone();
            size[axis] = groupSize;

            for (int i = 0; i < numGroups; i++)
            {
                begin[axis] = i * groupSize;
                result[i] = this.Slice(begin, size);
            }

            return result;
        }

        /// <summary>
        /// Splits the tensor into multiple tensors along the specified axis, with specified sizes.
        /// </summary>
        /// <param name="sizes">An array specifying the size of each split along the given axis.</param>
        /// <param name="axis">The axis along which to split the tensor.</param>
        /// <returns>An array of tensors resulting from the split.</returns>
        public Tensor[] Split(int[] sizes, int axis = 0)
        {
            if (axis < 0 || axis >= this.Shape.Length)
            {
                throw new ArgumentException("Axis is out of bounds for the tensor.");
            }

            if (sizes.Sum() != this.Shape[axis])
            {
                throw new ArgumentException("The sum of sizes must equal the dimension of the axis being split.");
            }

            Tensor[] result = new Tensor[sizes.Length];
            int[] begin = new int[this.Shape.Length];
            int[] size = (int[])this.Shape.Clone();

            int startIndex = 0;
            for (int i = 0; i < sizes.Length; i++)
            {
                begin[axis] = startIndex;
                size[axis] = sizes[i];
                result[i] = this.Slice(begin, size);
                startIndex += sizes[i];
            }

            return result;
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
        /// Transposes the tensor according to the specified permutation of axes.
        /// </summary>
        /// <param name="permutation">The permutation of the axes.</param>
        /// <returns>A new tensor that is the transposed version of the original tensor.</returns>
        /// <exception cref="ArgumentException">If the permutation is invalid.</exception>
        public Tensor Transpose(params int[] permutation)
        {
            if (permutation.Length != this.Shape.Length)
            {
                throw new ArgumentException("The permutation must have the same length as the number of dimensions of the tensor.");
            }

            if (permutation.Distinct().Count() != permutation.Length || permutation.Any(p => p < 0 || p >= this.Shape.Length))
            {
                throw new ArgumentException("The permutation must be a valid permutation of the dimensions.");
            }

            // Calculate the new shape after transposition
            var newShape = permutation.Select(p => this.Shape[p]).ToArray();
            var result = new Tensor(newShape);

            // Calculate strides for both the original and transposed tensors
            var originalStrides = CalculateStrides(this.Shape);
            var transposedStrides = CalculateStrides(newShape);

            // Perform the transposition
            Parallel.For(0, result.Data.Length, i =>
            {
                int[] newIndices = this.GetMultiDimensionalIndices(i, newShape);
                int[] oldIndices = new int[newIndices.Length];
                for (int j = 0; j < newIndices.Length; j++)
                {
                    oldIndices[permutation[j]] = newIndices[j];
                }

                int oldIndex = this.GetIndex(oldIndices);
                result.Data[i] = this.Data[oldIndex];
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

        /// <summary>
        /// An indexer for the tensor.
        /// </summary>
        /// <param name="indices">The indices used to slice.</param>
        /// <returns>The sliced tensor.</returns>
        /// <exception cref="ArgumentException">Number of indices does not match rank.</exception>
        public Tensor Indexer(params string[] indices)
        {
            if (indices.Length != this.Shape.Length)
            {
                throw new ArgumentException($"Number of indices ({indices.Length}) does not match tensor rank ({this.Shape.Length})");
            }

            int[] start = new int[this.Shape.Length];
            int[] end = new int[this.Shape.Length];
            int[] step = new int[this.Shape.Length];
            bool[] isSlice = new bool[this.Shape.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                this.ParseIndex(indices[i], this.Shape[i], out start[i], out end[i], out step[i], out isSlice[i]);
            }

            int[] newShape = this.CalculateNewShape(start, end, step, isSlice);
            Tensor result = new Tensor(newShape);

            this.CopyData(this, result, start, end, step, isSlice, new int[this.Shape.Length], new int[newShape.Length], 0);

            return result;
        }

        /// <summary>
        /// Calculates the number of slices for each dimension based on the specified slice sizes.
        /// </summary>
        /// <param name="sliceSizes">The sizes of each slice for each dimension.</param>
        /// <returns>The number of slices for each dimension.</returns>
        public int[] CalculateNumberOfSlices(int[] sliceSizes)
        {
            if (sliceSizes.Length != this.Shape.Length)
            {
                throw new ArgumentException("Slice sizes length must match the tensor shape length.");
            }

            var numSlices = new int[this.Shape.Length];
            for (int i = 0; i < this.Shape.Length; i++)
            {
                numSlices[i] = (int)Math.Ceiling((double)this.Shape[i] / sliceSizes[i]);
            }

            return numSlices;
        }

        /// <summary>
        /// Generates all possible slice start indices based on the number of slices for each dimension.
        /// </summary>
        /// <param name="numSlices">The number of slices for each dimension.</param>
        /// <returns>A list of slice start indices for each dimension.</returns>
        public List<int[]> GenerateSliceStartIndices(int[] numSlices)
        {
            var indicesList = new List<int[]>();
            this.GenerateIndicesRecursively(new int[numSlices.Length], 0, numSlices, indicesList);
            return indicesList;
        }

        /// <summary>
        /// Adjusts a negative index to a positive index based on the maximum rank.
        /// </summary>
        /// <param name="index">The index to adjust.</param>
        /// <param name="maxRank">The maximum rank of the tensor.</param>
        /// <returns>The adjusted index.</returns>
        internal static int AdjustNegativeIndex(int index, int maxRank)
        {
            return index < 0 ? maxRank + index : index;
        }

        /// <summary>
        /// Computes the strides for each dimension of the tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>An array of strides for each dimension.</returns>
        internal static int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }

        /// <summary>
        /// Parses the axis range string and returns the start and end values.
        /// </summary>
        /// <param name="axisRange">The axis range.</param>
        /// <returns>The start and end values.</returns>
        /// <exception cref="ArgumentException">Invalid axis range format.</exception>
        internal static (int Start, int End) ParseAxisRange(string axisRange)
        {
            var parts = axisRange.Split(':');
            if (parts.Length != 2 || !int.TryParse(parts[0], out var start) || !int.TryParse(parts[1], out var end))
            {
                throw new ArgumentException("Invalid axis range format. Use 'start:end'.");
            }

            return (start, end);
        }

        /// <summary>
        /// Determines the scenario based on the start, end, and concatAxis.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <param name="concatAxis">The concatenation axis.</param>
        /// <param name="maxRank">The maximum rank.</param>
        /// <returns>The scenario.</returns>
        internal static int DetermineScenario(int start, int end, int concatAxis, int maxRank)
        {
            if (start == 0 && end == maxRank && (concatAxis == 0 || concatAxis == 1))
            {
                return 1; // Standard concatenation
            }

            if (start == maxRank - 2 && end == maxRank)
            {
                return concatAxis switch
                {
                    0 => 5,
                    1 => 6,
                    2 => 7,
                    _ => 0
                };
            }

            if (start == 1 && end == 3)
            {
                return concatAxis switch
                {
                    0 => 2,
                    1 => 3,
                    2 => 4,
                    _ => 0
                };
            }

            return 0; // Unknown scenario
        }

        /// <summary>
        /// Gets the index based on the indices.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>The index.</returns>
        /// <exception cref="ArgumentException">Indices length invalid.</exception>
        internal int GetIndex(int[] indices)
        {
            if (indices.Length != this.Shape.Length)
            {
                throw new ArgumentException("Indices length does not match tensor shape.");
            }

            int index = 0;
            int stride = 1;
            for (int i = this.Shape.Length - 1; i >= 0; i--)
            {
                index += indices[i] * stride;
                stride *= this.Shape[i];
            }

            return index;
        }

        /// <summary>
        /// Gets the multi-dimensional indices.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="shape">The shape.</param>
        /// <returns>The multi-dimensional indices.</returns>
        internal int[] GetMultiDimensionalIndices(int index, int[] shape)
        {
            int[] indices = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = (index / stride) % shape[i];
                stride *= shape[i];
            }

            return indices;
        }

        /// <summary>
        /// Parse a single index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="dimSize">The dimension size.</param>
        /// <returns>The single index.</returns>
        internal int ParseSingleIndex(string index, int dimSize)
        {
            int result = int.Parse(index);
            return result < 0 ? dimSize + result : result;
        }

        /// <summary>
        /// Parse an index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="dimSize">The dimension size.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <param name="step">The step size.</param>
        /// <param name="isSlice">Is slice.</param>
        /// <exception cref="ArgumentException">Step is zero.</exception>
        internal void ParseIndex(string index, int dimSize, out int start, out int end, out int step, out bool isSlice)
        {
            isSlice = index.Contains(':') || index == "...";
            step = 1;
            start = 0;
            end = dimSize;

            if (index == "...")
            {
                return;
            }

            string[] parts = index.Split(':');

            if (parts.Length == 1)
            {
                // Single index
                start = end = this.ParseSingleIndex(parts[0], dimSize);
                isSlice = false;
            }
            else if (parts.Length == 2)
            {
                // Start:End
                start = string.IsNullOrEmpty(parts[0]) ? 0 : this.ParseSingleIndex(parts[0], dimSize);
                end = string.IsNullOrEmpty(parts[1]) ? dimSize : this.ParseSingleIndex(parts[1], dimSize);
            }
            else if (parts.Length == 3)
            {
                // Start:End:Step
                start = string.IsNullOrEmpty(parts[0]) ? 0 : this.ParseSingleIndex(parts[0], dimSize);
                end = string.IsNullOrEmpty(parts[1]) ? dimSize : this.ParseSingleIndex(parts[1], dimSize);
                step = string.IsNullOrEmpty(parts[2]) ? 1 : int.Parse(parts[2]);
            }

            if (step == 0)
            {
                throw new ArgumentException("Step cannot be zero");
            }

            if (step < 0)
            {
                if (start == 0)
                {
                    start = dimSize - 1;
                }

                if (end == dimSize)
                {
                    end = -1;
                }
            }

            // Adjust negative indices
            start = start < 0 ? dimSize + start : start;
            end = end < 0 ? dimSize + end : end;

            // Clamp to valid range
            start = Math.Max(0, Math.Min(start, dimSize - 1));
            end = Math.Max(0, Math.Min(end, dimSize));
        }

        /// <summary>
        /// Calculate the new shape.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <param name="step">The step size.</param>
        /// <param name="isSlice">Is slice.</param>
        /// <returns>The new shape.</returns>
        internal int[] CalculateNewShape(int[] start, int[] end, int[] step, bool[] isSlice)
        {
            List<int> newShape = new List<int>();
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (isSlice[i])
                {
                    int size = (int)Math.Ceiling((double)(end[i] - start[i]) / step[i]);
                    newShape.Add(size);
                }
            }

            return newShape.ToArray();
        }

        /// <summary>
        /// Concatenates tensors along a specific axis.
        /// </summary>
        private static Tensor StandardConcat(Tensor[] tensors, int concatAxis)
        {
            return Tensor.Concat(tensors, concatAxis);
        }

        /// <summary>
        /// Concatenates tensors by unstacking along a specified axis, expanding dimensions, and concatenating.
        /// </summary>
        private static Tensor ConcatByUnstackAndExpand(Tensor[] tensors, int unstackAxis, int concatAxis, bool expandDim = false, bool expandDimAtEnd = false)
        {
            var maxRank = tensors.Max(t => t.Shape.Length);
            var allSlices = new List<Tensor>();
            foreach (var tensor in tensors)
            {
                if (tensor.Shape.Length == maxRank)
                {
                    var unstackedSlices = tensor.Unstack(unstackAxis);
                    foreach (var slice in unstackedSlices)
                    {
                        allSlices.Add(expandDim ? slice.ExpandDims(0) : slice);
                    }
                }
                else
                {
                    allSlices.Add(expandDim ? tensor.ExpandDims(0) : tensor);
                }
            }

            var result = Tensor.Concat(allSlices.ToArray(), concatAxis);

            if (expandDimAtEnd)
            {
                result = result.ExpandDims(0);
            }

            return result;
        }

        /// <summary>
        /// Concatenates tensors by expanding them to match the maximum rank.
        /// </summary>
        private static Tensor ConcatByExpansion(Tensor[] tensors, int concatAxis)
        {
            var allSlices = new List<Tensor>();
            foreach (var tensor in tensors)
            {
                if (tensor.Shape.Length == 2)
                {
                    allSlices.Add(tensor.ExpandDims(0));
                }
                else
                {
                    allSlices.Add(tensor);
                }
            }

            return Tensor.Concat(allSlices.ToArray(), concatAxis);
        }

        private static int[] CalculateStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            return strides;
        }

        private void GenerateIndicesRecursively(int[] currentIndex, int dim, int[] numSlices, List<int[]> indicesList)
        {
            if (dim == numSlices.Length)
            {
                indicesList.Add((int[])currentIndex.Clone());
                return;
            }

            for (int i = 0; i < numSlices[dim]; i++)
            {
                currentIndex[dim] = i;
                this.GenerateIndicesRecursively(currentIndex, dim + 1, numSlices, indicesList);
            }
        }

        private void CopyData(Tensor source, Tensor dest, int[] start, int[] end, int[] step, bool[] isSlice, int[] sourceIndices, int[] destIndices, int currentDim)
        {
            if (currentDim == source.Shape.Length)
            {
                dest[destIndices] = source[sourceIndices];
                return;
            }

            int sourceStart = start[currentDim];
            int sourceEnd = end[currentDim];
            int sourceStep = step[currentDim];

            if (isSlice[currentDim])
            {
                int destIndex = 0;
                for (int i = sourceStart; i < sourceEnd; i += sourceStep)
                {
                    int[] newSourceIndices = (int[])sourceIndices.Clone();
                    newSourceIndices[currentDim] = i;

                    int[] newDestIndices = (int[])destIndices.Clone();
                    newDestIndices[currentDim] = destIndex;

                    this.CopyData(source, dest, start, end, step, isSlice, newSourceIndices, newDestIndices, currentDim + 1);
                    destIndex++;
                }
            }
            else
            {
                int[] newSourceIndices = (int[])sourceIndices.Clone();
                newSourceIndices[currentDim] = sourceStart;

                this.CopyData(source, dest, start, end, step, isSlice, newSourceIndices, destIndices, currentDim + 1);
            }
        }

        private int GetTotalSize(int[] shape)
        {
            int total = 1;
            foreach (var dim in shape)
            {
                total *= dim;
            }

            return total;
        }

        private void CheckShapeCompatibility(Tensor other)
        {
            if (this.Shape.Length != other.Shape.Length)
            {
                throw new ArgumentException("Shapes are not compatible for the operation.");
            }

            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (this.Shape[i] != other.Shape[i])
                {
                    throw new ArgumentException("Shapes are not compatible for the operation.");
                }
            }
        }

        private void SumRecursive(int[] inputIndex, int[] resultIndex, int currentAxis, ref double sum, int[] strides)
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

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
        /// Gets the data for the tensor.
        /// </summary>
        public double[] Data { get; private set; }

        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        public int[] Shape { get; private set; }

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

            // Ensure all tensors have the same shape except along the concatenation axis
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

            var totalSize = outputShape.Aggregate((a, b) => a * b);
            var outputData = new double[totalSize];
            var outputTensor = new Tensor(outputShape, outputData);

            // Concatenate the data
            int offset = 0;
            foreach (var tensor in tensors)
            {
                int tensorSize = tensor.Data.Length;
                Buffer.BlockCopy(tensor.Data, 0, outputData, offset * sizeof(double), tensorSize * sizeof(double));
                offset += tensorSize;
            }

            return outputTensor;
        }

        /// <summary>
        /// Gathers slices from the tensor along the specified axis.
        /// </summary>
        /// <param name="indices">The indices of elements to gather.</param>
        /// <param name="axis">The axis along which to gather slices.</param>
        /// <param name="batchDims">The number of batch dimensions.</param>
        /// <returns>A new tensor with the gathered slices.</returns>
        public Tensor Gather(Tensor indices, int axis = 0, int batchDims = 0)
        {
            if (batchDims < 0 || batchDims > axis || batchDims >= this.Shape.Length || batchDims >= indices.Shape.Length)
            {
                throw new ArgumentException("Invalid batch dimensions.");
            }

            // Handle negative indices
            var indicesData = indices.Data.Select(i => i < 0 ? this.Shape[axis] + (int)i : (int)i).ToArray();

            // Validate indices
            foreach (var index in indicesData)
            {
                if (index < 0 || index >= this.Shape[axis])
                {
                    throw new ArgumentException("Index out of bounds.");
                }
            }

            // Prepare the shape of the result tensor
            var resultShape = new int[indices.Shape.Length + this.Shape.Length - 1];
            for (int i = 0; i < axis; i++)
            {
                resultShape[i] = this.Shape[i];
            }

            resultShape[axis] = indices.Shape[axis];
            for (int i = axis + 1; i < this.Shape.Length; i++)
            {
                resultShape[i + indices.Shape.Length - 1] = this.Shape[i];
            }

            var result = new Tensor(resultShape);

            // Gather data
            Parallel.For(0, indicesData.Length, i =>
            {
                int[] indicesForGather = new int[this.Shape.Length];
                for (int j = 0; j < this.Shape.Length; j++)
                {
                    if (j == axis)
                    {
                        indicesForGather[j] = indicesData[i];
                    }
                    else
                    {
                        indicesForGather[j] = i;
                    }
                }

                result.Data[i] = this[indicesForGather];
            });

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
                    for (int i = 0; i < multiples[dim - 1]; i++)
                    {
                        Array.Copy(indices, resultIndices, dim);
                        resultIndices[dim - 1] += i * this.Shape[dim - 1];
                        result[resultIndices] = value;
                    }
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
        /// Computes the element-wise square of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the result.</returns>
        public Tensor ElementwiseSquare()
        {
            var result = new Tensor(this.Shape);
            Vml.Sqr(this.Data.Length, this.Data, result.Data);
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
        /// <param name="batchDims">The number of batch dimensions.</param>
        /// <returns>A new tensor with the gathered slices.</returns>
        public Tensor GatherNd(Tensor indices, int batchDims = 0)
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
        /// Extracts a slice from the tensor based on begin, size, and optional strides.
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

            strides = strides ?? new int[this.Shape.Length];
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (strides[i] == 0)
                {
                    strides[i] = 1;
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
                    size[i] = this.Shape[i] - begin[i];
                }

                if (begin[i] + (size[i] * strides[i]) > this.Shape[i])
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
        /// Creates a flat array from the tensors along the specified indices.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>The flat array.</returns>
        public Tensor CreateFlatArray(Tensor[] tensors, int[] indices)
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
        /// Splits the tensor into multiple tensors along the specified axis.
        /// </summary>
        /// <param name="groupSize">The group size.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The tensors.</returns>
        public Tensor[] Split(int groupSize, int axis = 0)
        {
            int numGroups = this.Shape[axis] / groupSize;
            Tensor[] result = new Tensor[numGroups];

            for (int i = 0; i < numGroups; i++)
            {
                int start = i * groupSize;
                result[i] = this.Slice(new int[] { start }, new int[] { groupSize }, new int[] { axis });
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

        private int GetTotalSize(int[] shape)
        {
            int total = 1;
            foreach (var dim in shape)
            {
                total *= dim;
            }

            return total;
        }

        private int GetIndex(int[] indices)
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
    }
}

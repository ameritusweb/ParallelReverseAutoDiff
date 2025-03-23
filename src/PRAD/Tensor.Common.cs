//------------------------------------------------------------------------------
// <copyright file="Tensor.Common.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using MKLNET;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A flat tensor.
    /// </summary>
    public partial class Tensor
    {
        private static readonly ThreadLocal<Random> RandomGen = new ThreadLocal<Random>(() =>
        {
            DateTime today = DateTime.UtcNow.Date;
            int baseSeed = (today.Year * 10000) + (today.Month * 100) + today.Day;
            int threadOffset = Thread.CurrentThread.ManagedThreadId;
            return new Random(baseSeed + threadOffset);
        });

        private int[] strides;

        /// <summary>
        /// Gets the strides.
        /// </summary>
        public int[] Strides
        {
            get
            {
                if (this.strides == null)
                {
                    this.strides = new int[this.Shape.Length];
                    this.strides[this.strides.Length - 1] = 1;
                    for (int i = this.strides.Length - 2; i >= 0; i--)
                    {
                        this.strides[i] = this.strides[i + 1] * this.Shape[i + 1];
                    }
                }

                return this.strides;
            }
        }

        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        public int[] Shape { get; private set; }

        /// <summary>
        /// Gets the debug info.
        /// </summary>
        public string DebugInfo { get; private set; }

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
        /// Returns a tensor formed by selecting elements from two input tensors, based on a condition tensor.
        /// </summary>
        /// <param name="condition">A tensor containing boolean-like values (0 or 1), where 1 indicates that the corresponding element should be taken from tensor <paramref name="x"/> and 0 indicates that the corresponding element should be taken from tensor <paramref name="y"/>.</param>
        /// <param name="x">The tensor from which to select elements when the condition is true (1).</param>
        /// <param name="y">The tensor from which to select elements when the condition is false (0).</param>
        /// <returns>A new tensor with the same shape as the input tensors, containing elements selected from <paramref name="x"/> and <paramref name="y"/> based on the <paramref name="condition"/> tensor.</returns>
        /// <exception cref="ArgumentException">Thrown if the shapes of the input tensors do not match.</exception>
        /// <remarks>
        /// This method uses SIMD (Single Instruction, Multiple Data) vectorization to optimize the element selection process for large tensors.
        /// The method processes elements in blocks of vector size, then handles any remaining elements that do not fit into a full vector.
        /// </remarks>
        public static Tensor Where(Tensor condition, Tensor x, Tensor y)
        {
            if (!condition.Shape.SequenceEqual(x.Shape) || !condition.Shape.SequenceEqual(y.Shape))
            {
                throw new ArgumentException("All tensors must have the same shape for the Where operation.");
            }

            var result = new Tensor(x.Shape);
            int vectorSize = PradTools.VectorCount();

            for (int i = 0; i <= condition.Data.Length - vectorSize; i += vectorSize)
            {
                var conditionVector = PradTools.AllocateVector(condition.Data, i);
                var xVector = PradTools.AllocateVector(x.Data, i);
                var yVector = PradTools.AllocateVector(y.Data, i);
                var resultVector = Vector.ConditionalSelect(
                    Vector.Equals(conditionVector, PradTools.AllocateVector(PradTools.One)),
                    xVector,
                    yVector);
                resultVector.CopyTo(result.Data, i);
            }

            // Handle remaining elements
            for (int i = condition.Data.Length - (condition.Data.Length % vectorSize); i < condition.Data.Length; i++)
            {
                result.Data[i] = condition.Data[i] != 0 ? x.Data[i] : y.Data[i];
            }

            return result;
        }

        /// <summary>
        /// Concatenates a list of tensors along a specified axis, based on a custom ordering of the tensors.
        /// If the ordering is null, the tensors will be concatenated in their original order.
        /// </summary>
        /// <param name="tensors">The list of tensors to concatenate.</param>
        /// <param name="axis">The axis along which to concatenate the tensors. Defaults to 0.</param>
        /// <param name="ordering">An optional array representing the ordering of the tensors by index. If null, no reordering is done.</param>
        /// <returns>A new concatenated tensor.</returns>
        /// <exception cref="ArgumentException">Thrown if tensors list is null, fewer than two tensors, invalid axis, or incompatible tensor shapes.</exception>
        public static Tensor Concat(Tensor[] tensors, int axis = 0, int[]? ordering = null)
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

            // Validate axis bounds
            if (axis < 0 || axis >= rank)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Reorder tensors based on the provided ordering, or use the original tensors if ordering is null
            Tensor[] tensorsToConcat;
            if (ordering != null)
            {
                if (ordering.Length != tensors.Length)
                {
                    throw new ArgumentException("The ordering array must be the same length as the tensors array.");
                }

                tensorsToConcat = new Tensor[tensors.Length];
                for (int i = 0; i < ordering.Length; i++)
                {
                    if (ordering[i] < 0 || ordering[i] >= tensors.Length)
                    {
                        throw new ArgumentException("Invalid ordering index.");
                    }

                    tensorsToConcat[i] = tensors[ordering[i]];
                }
            }
            else
            {
                tensorsToConcat = tensors;
            }

            // Validate tensor ranks and calculate the total size along the concatenation axis
            int concatDimSize = 0;
            foreach (var tensor in tensorsToConcat)
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

            // Determine the output tensor shape
            var outputShape = new int[rank];
            Array.Copy(shape, outputShape, rank);
            outputShape[axis] = concatDimSize;

            var outputData = PradTools.AllocateArray(outputShape.Aggregate(1, (a, b) => a * b));
            var outputTensor = new Tensor(outputShape, outputData);

            // Calculate slice size and number of slices
            int sliceSize = 1;
            for (int i = axis + 1; i < rank; i++)
            {
                sliceSize *= shape[i];
            }

            int numSlices = 1;
            for (int i = 0; i < axis; i++)
            {
                numSlices *= shape[i];
            }

            // Concatenate the tensors (in original order or reordered based on `ordering`)
            int outputOffset = 0;
            for (int slice = 0; slice < numSlices; slice++)
            {
                foreach (var tensor in tensorsToConcat)
                {
                    int tensorSliceSize = sliceSize * tensor.Shape[axis];
                    int inputOffset = slice * tensorSliceSize;
                    Buffer.BlockCopy(tensor.Data, inputOffset * PradTools.SizeOf, outputData, outputOffset * PradTools.SizeOf, tensorSliceSize * PradTools.SizeOf);
                    outputOffset += tensorSliceSize;
                }
            }

            return outputTensor;
        }

        /// <summary>
        /// Fills a tensor with normally distributed random numbers using the Box-Muller transform.
        /// </summary>
        /// <param name="shape">The shape of the resulting tensor.</param>
        /// <returns>A tensor filled with normally distributed random numbers.</returns>
        public static Tensor RandomNormal(int[] shape)
        {
            // Create the result tensor with the specified shape
            var result = new Tensor(shape);

            // Total number of elements to fill
            int totalSize = result.Data.Length;

            // Random number generator
            Random rand = new Random();

            // Generate normally distributed numbers in pairs (Box-Muller generates two at a time)
            for (int i = 0; i < totalSize; i += 2)
            {
                // Generate two uniformly distributed random values u1 and u2
                var u1 = PradTools.One - PradTools.Cast(rand.NextDouble()); // avoid log(0) by subtracting from 1
                var u2 = PradTools.One - PradTools.Cast(rand.NextDouble());

                // Apply Box-Muller transform to get two independent normally distributed numbers
                var r = PradMath.Sqrt(PradTools.NegativeTwo * PradMath.Log(u1));
                var z0 = r * PradMath.Cos(PradTools.Two * PradMath.PI * u2);
                var z1 = r * PradMath.Sin(PradTools.Two * PradMath.PI * u2);

                // Assign the first random number to the tensor
                result.Data[i] = z0;

                // Assign the second random number if within bounds (since we process two numbers at a time)
                if (i + 1 < totalSize)
                {
                    result.Data[i + 1] = z1;
                }
            }

            return result;
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
        /// Initializes a tensor using the Xavier (Glorot) uniform initialization.
        /// </summary>
        /// <param name="shape">The shape of the tensor to initialize.</param>
        /// <returns>A tensor initialized with values drawn from the Xavier uniform distribution.</returns>
        public static Tensor XavierUniform(int[] shape)
        {
            if (shape.Length < 2)
            {
                throw new ArgumentException("Xavier initialization requires at least two dimensions (e.g., for weights in a neural network layer).");
            }

            // Calculate fan_in and fan_out
            int fanIn = shape[shape.Length - 2];
            int fanOut = shape[shape.Length - 1];

            // Xavier uniform range
            var limit = PradMath.Sqrt(PradTools.Six / (fanIn + fanOut));

            // Create a tensor and fill it with values from a uniform distribution in the range [-limit, limit]
            var result = new Tensor(shape);
            int totalSize = result.Data.Length;

            var randomArray = GenerateRandomDoubleArray(totalSize);

            // Thread-safe random number generator
            Parallel.For(0, totalSize, i =>
            {
                // Generate a uniform random value in [-1, 1]
                var randomValue = PradTools.Cast((2.0 * randomArray[i]) - 1.0);

                // Scale to the range [-limit, limit]
                result.Data[i] = randomValue * limit;
            });

            return result;
        }

        /// <summary>
        /// Initializes a tensor using the Xavier (Glorot) normal initialization.
        /// </summary>
        /// <param name="shape">The shape of the tensor to initialize.</param>
        /// <returns>A tensor initialized with values drawn from the Xavier normal distribution.</returns>
        public static Tensor XavierNormal(int[] shape)
        {
            if (shape.Length < 2)
            {
                throw new ArgumentException("Xavier initialization requires at least two dimensions (e.g., for weights in a neural network layer).");
            }

            // Calculate fan_in and fan_out
            int fanIn = shape[shape.Length - 2];
            int fanOut = shape[shape.Length - 1];

            // Xavier normal standard deviation
            var stddev = PradMath.Sqrt(PradTools.Two / (fanIn + fanOut));

            // Create a tensor and fill it with normally distributed values
            var result = new Tensor(shape);

            // Use Box-Muller transform to generate normally distributed values
            int totalSize = result.Data.Length;
            Parallel.For(0, totalSize / 2, i =>
            {
                // Generate two uniformly distributed random values u1 and u2
                var u1 = (float)RandomGen.Value.NextDouble();
                var u2 = (float)RandomGen.Value.NextDouble();

                // Box-Muller transform to generate two independent standard normal values
                var r = PradMath.Sqrt(PradTools.NegativeTwo * PradMath.Log(u1));
                var theta = PradTools.Two * PradMath.PI * u2;

                var z0 = r * PradMath.Cos(theta);
                var z1 = r * PradMath.Sin(theta);

                // Apply Xavier scaling (stddev)
                result.Data[2 * i] = z0 * stddev;
                if ((2 * i) + 1 < totalSize)
                {
                    result.Data[(2 * i) + 1] = z1 * stddev;
                }
            });

            return result;
        }

        /// <summary>
        /// Adds noise to the tensor, with the noise mean equal to the mean of the tensor's distribution,
        /// and the standard deviation scaled by the provided scale factor. The dropout parameter controls
        /// the fraction of elements that will remain unchanged.
        /// </summary>
        /// <param name="tensor">The tensor to which noise will be applied.</param>
        /// <param name="scale">The scale factor for the noise variance. A higher scale increases the variance of the added noise.</param>
        /// <param name="dropoutProbability">The probability of dropping out noise for each element. 0 means no dropout, 1 means full dropout.</param>
        /// <returns>A new tensor with added noise and dropout applied.</returns>
        public static Tensor ApplyNoise(Tensor tensor, double scale, double dropoutProbability)
        {
            if (scale <= 0)
            {
                throw new ArgumentException("Scale must be a positive value.");
            }

            if (dropoutProbability < 0 || dropoutProbability > 1)
            {
                throw new ArgumentException("Dropout probability must be between 0 and 1.");
            }

            // Calculate the mean of the input tensor
            var mean = tensor.Data.Average();

            // Calculate the standard deviation of the input tensor
            var variance = tensor.Data.Select(x => PradMath.Pow(x - mean, 2)).Sum() / tensor.Data.Length;
            var stddev = PradMath.Sqrt(variance);

            // Adjust the standard deviation by the scale factor
            var scaledStddev = stddev * (float)scale;

            // Create a new tensor to hold the result
            var result = new Tensor(tensor.Shape);

            // Apply noise with dropout to each element
            Parallel.For(0, tensor.Data.Length, i =>
            {
                // Generate a dropout decision
                if (RandomGen.Value.NextDouble() > dropoutProbability)
                {
                    // Apply noise (Gaussian noise with mean = 0 and stddev = 1)
                    var u1 = (float)RandomGen.Value.NextDouble();
                    var u2 = (float)RandomGen.Value.NextDouble();
                    var noise = PradMath.Sqrt(PradTools.NegativeTwo * PradMath.Log(u1)) * PradMath.Cos(PradTools.Two * PradMath.PI * u2);

                    // Scale the noise and add it to the original tensor value
                    result.Data[i] = tensor.Data[i] + (mean + (noise * scaledStddev));
                }
                else
                {
                    // No noise is applied (dropout), keep the original value
                    result.Data[i] = tensor.Data[i];
                }
            });

            return result;
        }

        /// <summary>
        /// Generates unique pairs within a single 1D tensor without repetition or self-pairing.
        /// Optimized with precomputed offsets and parallelism.
        /// </summary>
        /// <returns>A tensor of shape [2, M] where each column represents a unique pair.</returns>
        public Tensor SelfPair()
        {
            var tensor = this;

            if (tensor.Shape.Length != 2 || tensor.Shape[0] != 1)
            {
                throw new ArgumentException("Input tensor must be of shape [1, N].");
            }

            int n = tensor.Shape[1];
            int m = n * (n - 1) / 2; // Number of unique pairs

            // Create the output tensor of shape [2, M]
            var resultShape = new int[] { 2, m };
            var result = new Tensor(resultShape);

            // Calculate offset for each row based on cumulative pair count up to i
            int[] offsets = new int[n];
            for (int i = 1; i < n; i++)
            {
                offsets[i] = offsets[i - 1] + (n - i);
            }

            // Generate unique pairs using the precomputed offsets
            Parallel.For(0, n - 1, i =>
            {
                int offset = offsets[i];
                for (int j = i + 1; j < n; j++)
                {
                    result.Data[offset] = tensor.Data[i];       // First row value
                    result.Data[m + offset] = tensor.Data[j];   // Second row value
                    offset++;
                }
            });

            return result;
        }

        /// <summary>
        /// Multiplies the values together in each column of the tensor.
        /// </summary>
        /// <returns>A tensor of shape [1, P] where each element is the product of the corresponding column in the original tensor.</returns>
        public Tensor MultiplyColumns()
        {
            if (this.Shape.Length != 2)
            {
                throw new ArgumentException("MultiplyColumns operation is only defined for 2D tensors.");
            }

            int rows = this.Shape[0];
            int columns = this.Shape[1];

            // Create the output tensor of shape [1, columns]
            var result = new Tensor(new int[] { 1, columns });

            // Compute the product of each column in parallel
            Parallel.For(0, columns, col =>
            {
                var product = PradTools.One;
                for (int row = 0; row < rows; row++)
                {
                    product *= this.Data[(row * columns) + col];
                }

                result.Data[col] = product;
            });

            return result;
        }

        /// <summary>
        /// Computes the element-wise arc cosine (inverse cosine) of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the element-wise arc cosine values.</returns>
        public Tensor ElementwiseArcCos()
        {
            var result = new Tensor(this.Shape); // Create a new tensor with the same shape as the current one.

            // Use MKL.NET VML to compute the arc cosine of each element in the tensor.
            Vml.Acos(this.Data.Length, this.Data, result.Data);

            return result; // Return the resultant tensor containing the arc cosine values.
        }

        /// <summary>
        /// Computes the element-wise arc cosine (inverse cosine) of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the element-wise arc cosine values.</returns>
        public Tensor ElementwiseArcSin()
        {
            var result = new Tensor(this.Shape); // Create a new tensor with the same shape as the current one.

            // Use MKL.NET VML to compute the arc cosine of each element in the tensor.
            Vml.Asin(this.Data.Length, this.Data, result.Data);

            return result; // Return the resultant tensor containing the arc cosine values.
        }

        /// <summary>
        /// Performs an embedding lookup on the current tensor based on provided embeddings.
        /// Supports 1D, 2D, and 3D tensors with optional batch dimensions for embeddings.
        /// </summary>
        /// <param name="embeddings">The 2D or 3D tensor containing embedding vectors.</param>
        /// <returns>A new tensor with the embedded vectors.</returns>
        /// <exception cref="ArgumentException">Thrown when the tensor shapes are invalid.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when an index is out of the embedding range.</exception>
        public Tensor Embedding(Tensor embeddings)
        {
            // Validate the index tensor (this)
            if (this.Shape.Length < 1 || this.Shape.Length > 3)
            {
                throw new ArgumentException("The index tensor must be 1D, 2D, or 3D.");
            }

            // Validate the embeddings tensor
            if (embeddings.Shape.Length != 2 && embeddings.Shape.Length != 3)
            {
                throw new ArgumentException("The embeddings tensor must be 2D or 3D.");
            }

            // Check if embeddings are batched
            bool batchedEmbeddings = embeddings.Shape.Length == 3;

            // Determine number of embeddings and embedding dimension
            int numEmbeddings = batchedEmbeddings ? embeddings.Shape[1] : embeddings.Shape[0];
            int embeddingDim = batchedEmbeddings ? embeddings.Shape[2] : embeddings.Shape[1];

            // Compute the output shape based on the index tensor's shape
            int[] outputShape = this.Shape.Length switch
            {
                1 => new int[] { this.Shape[0], embeddingDim },
                2 => new int[] { this.Shape[0] * this.Shape[1], embeddingDim },
                3 => new int[] { this.Shape[0], this.Shape[1] * this.Shape[2], embeddingDim },
                _ => throw new ArgumentException("Unexpected tensor shape.") // Safety fallback
            };

            // Create output tensor
            var result = new Tensor(outputShape);

            // Check if all indices are in range first to avoid partial failure
            for (int i = 0; i < this.Data.Length; i++)
            {
                int index = (int)this.Data[i];
                if (index < 0 || index >= numEmbeddings)
                {
                    throw new ArgumentOutOfRangeException($"Index {index} is out of range for embeddings with {numEmbeddings} entries.");
                }
            }

            // Perform the embedding lookup in parallel
            Parallel.For(0, this.Data.Length, i =>
            {
                int index = (int)this.Data[i];

                // If embeddings are batched, calculate the batch offset
                int batchIndex = batchedEmbeddings ? i / (this.Shape[^2] * this.Shape[^1]) : 0;

                int srcOffset = batchedEmbeddings
                    ? ((batchIndex * numEmbeddings) + index) * embeddingDim
                    : index * embeddingDim;

                int destOffset = i * embeddingDim;

                // Copy the embedding vector to the result tensor
                Array.Copy(embeddings.Data, srcOffset, result.Data, destOffset, embeddingDim);
            });

            return result;
        }

        /// <summary>
        /// Creates an "On-Off" embedding using a learned sparsity tensor T.
        /// </summary>
        /// <param name="indices">Tensor of indices, representing which rows to select.</param>
        /// <param name="binaryCondition">Tensor with binary values (0 or 1) indicating the condition for each index.</param>
        /// <param name="sparsityTensor">A learned tensor of shape [1, N] providing sparsity values for embedding interleaving.</param>
        /// <returns>A new tensor with doubled column size, applying the alternating pattern based on binaryCondition and sparsityTensor.</returns>
        /// <exception cref="ArgumentException">Thrown if indices and binaryCondition shapes don't match, or if binaryCondition contains values other than 0 or 1, or if sparsityTensor shape is incompatible.</exception>
        public Tensor OnOffEmbedding(Tensor indices, Tensor binaryCondition, Tensor sparsityTensor)
        {
            // Validate input shapes
            if (!indices.Shape.SequenceEqual(binaryCondition.Shape))
            {
                throw new ArgumentException("Indices and binary condition tensors must have the same shape.");
            }

            if (sparsityTensor.Shape.Length != 2 || sparsityTensor.Shape[0] != 1 || sparsityTensor.Shape[1] != this.Shape[1])
            {
                throw new ArgumentException("Sparsity tensor must be of shape [1, N], where N matches the embedding size.");
            }

            int embeddingSize = this.Shape[1];  // Original embedding size
            int newEmbeddingSize = embeddingSize * 2; // Doubled embedding size for on-off pattern

            // Validate binaryCondition values
            if (binaryCondition.Data.Any(b => b != 0 && b != 1))
            {
                throw new ArgumentException("Binary condition tensor must only contain values of 0 or 1.");
            }

            // Create the output tensor with doubled column size
            var resultShape = indices.Shape.Concat(new[] { newEmbeddingSize }).ToArray();
            var result = new Tensor(resultShape);

            Parallel.For(0, indices.Data.Length, i =>
            {
                int index = (int)indices.Data[i];
                int binary = (int)binaryCondition.Data[i];

                // Get the embedding row and the sparsity row (from sparsityTensor)
                int embeddingRowStart = index * embeddingSize;
                int sparsityRowStart = 0;  // Single row, so always starts at 0 for sparsityTensor

                // Result row for doubled columns
                int resultRowStart = i * newEmbeddingSize;

                // Populate the result row based on the binary condition
                for (int j = 0; j < embeddingSize; j++)
                {
                    if (binary == 0)
                    {
                        // For binary 0, place embedding value in even indices, sparsity value in odd indices
                        result.Data[resultRowStart + (2 * j)] = this.Data[embeddingRowStart + j];
                        result.Data[resultRowStart + (2 * j) + 1] = sparsityTensor.Data[sparsityRowStart + j];
                    }
                    else
                    {
                        // For binary 1, place sparsity value in even indices, embedding value in odd indices
                        result.Data[resultRowStart + (2 * j)] = sparsityTensor.Data[sparsityRowStart + j];
                        result.Data[resultRowStart + (2 * j) + 1] = this.Data[embeddingRowStart + j];
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Performs a vectorized bit flip operation on the tensor.
        /// This effectively converts 1s to 0s and 0s to 1s in one pass.
        /// </summary>
        /// <returns>A new tensor with flipped bits.</returns>
        public Tensor VectorizedBitFlip()
        {
            var result = new Tensor(this.Shape);
            int vectorSize = PradTools.VectorCount();

            int i = 0;
            for (; i <= this.Data.Length - vectorSize; i += vectorSize)
            {
                var vector = PradTools.AllocateVector(this.Data, i);
                var oneVector = PradTools.AllocateVector(PradTools.One);
                var resultVector = oneVector - vector;
                resultVector.CopyTo(result.Data, i);
            }

            // Handle remaining elements
            for (; i < this.Data.Length; i++)
            {
                result.Data[i] = PradTools.One - this.Data[i];
            }

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
        /// Element-wise maximum between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The other tensor to compare.</param>
        /// <returns>A new tensor containing the element-wise maximum values.</returns>
        /// <exception cref="ArgumentException">Thrown if the tensors do not have the same shape.</exception>
        public Tensor Max(Tensor other)
        {
            // Ensure the shapes match
            if (!this.Shape.SequenceEqual(other.Shape))
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise maximum.");
            }

            // Allocate the result tensor
            var resultData = PradTools.AllocateArray(this.Data.Length);
            var resultTensor = new Tensor(this.Shape, resultData);

            // Perform the element-wise maximum using MKLNET
            Vml.MaxMag(this.Data.Length, this.Data, other.Data, resultData);

            return resultTensor;
        }

        /// <summary>
        /// Generates all possible pairings between two 1D tensors.
        /// Optimized using Array.Fill, Array.Copy, and Parallel.For for efficient memory operations and parallelism.
        /// </summary>
        /// <param name="other">A tensor of shape [1, P].</param>
        /// <returns>A tensor of shape [2, N * P] where each column represents a pairing between the tensors.</returns>
        public Tensor PairwiseTile(Tensor other)
        {
            var tensor1 = this;
            var tensor2 = other;

            if (tensor1.Shape.Length != 2 || tensor1.Shape[0] != 1)
            {
                throw new ArgumentException("First tensor must be of shape [1, N].");
            }

            if (tensor2.Shape.Length != 2 || tensor2.Shape[0] != 1)
            {
                throw new ArgumentException("Second tensor must be of shape [1, P].");
            }

            int n = tensor1.Shape[1];
            int p = tensor2.Shape[1];

            // Create the output tensor of shape [2, N * P]
            var resultShape = new int[] { 2, n * p };
            var result = new Tensor(resultShape);

            // First row: Use Array.Fill and then Array.Copy to optimize memory operations
            Parallel.For(0, n, i =>
            {
                // Fill the block of P elements with the current value of tensor1[i]
                int startIndex = i * p;
                Array.Fill(result.Data, tensor1.Data[i], startIndex, p);
            });

            // Second row: Repeat tensor2 contents N times using Array.Copy
            Parallel.For(0, n, i =>
            {
                // Copy the entire tensor2 array into the corresponding location in the result
                Array.Copy(tensor2.Data, 0, result.Data, (n * p) + (i * p), p);
            });

            return result;
        }

        /// <summary>
        /// Element-wise minimum between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The other tensor to compare.</param>
        /// <returns>A new tensor containing the element-wise minimum values.</returns>
        /// <exception cref="ArgumentException">Thrown if the tensors do not have the same shape.</exception>
        public Tensor Min(Tensor other)
        {
            // Ensure the shapes match
            if (!this.Shape.SequenceEqual(other.Shape))
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise minimum.");
            }

            // Allocate the result tensor
            var resultData = PradTools.AllocateArray(this.Data.Length);
            var resultTensor = new Tensor(this.Shape, resultData);

            // Perform the element-wise minimum using MKLNET
            Vml.MinMag(this.Data.Length, this.Data, other.Data, resultData);

            return resultTensor;
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
        /// Tiles the tensor by the specified multiples along each dimension.
        /// </summary>
        /// <param name="multiples">An array specifying the tiling factor along each dimension.</param>
        /// <returns>A new tiled Tensor instance.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of multiples doesn't match the tensor dimensions or when any multiple is non-positive.</exception>
        public Tensor Tile(int[] multiples)
        {
            // Validation
            if (multiples.Length != this.Shape.Length || multiples.Any(m => m <= 0))
            {
                throw new ArgumentException("Invalid multiples: Must match tensor dimensions and all values must be positive.");
            }

            // Compute the new shape and total multiplication factor
            int[] newShape = new int[this.Shape.Length];
            int totalMultiple = 1;

            checked
            {
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    newShape[i] = this.Shape[i] * multiples[i];
                    totalMultiple *= multiples[i];
                }
            }

            var result = new Tensor(newShape);

            // Fast path for tiling along a single dimension
            int maxMultiple = multiples.Max();
            int tilingDimension = Array.IndexOf(multiples, maxMultiple);

            if (multiples.Count(m => m > 1) == 1)
            {
                int copySize = this.Data.Length;
                Array.Copy(this.Data, 0, result.Data, 0, copySize);

                for (int i = 1; i < multiples[tilingDimension]; i++)
                {
                    Array.Copy(result.Data, 0, result.Data, i * copySize, copySize);
                }

                return result;
            }

            // Fast path for equal multiples along all dimensions
            if (multiples.All(m => m == multiples[0]))
            {
                if (multiples[0] == 1)
                {
                    Array.Copy(this.Data, result.Data, this.Data.Length);
                    return result;
                }

                int blockSize = this.Data.Length;
                for (int i = 1; i < totalMultiple; i++)
                {
                    Array.Copy(this.Data, 0, result.Data, i * blockSize, blockSize);
                }

                return result;
            }

            // General case: tile each dimension separately
            int currentRepetitions = 1;
            int currentBlockSize = this.Data.Length;

            for (int dim = this.Shape.Length - 1; dim >= 0; dim--)
            {
                if (multiples[dim] == 1)
                {
                    continue;
                }

                int nextBlockSize = currentBlockSize * multiples[dim];

                for (int rep = 0; rep < currentRepetitions; rep++)
                {
                    int destOffset = rep * nextBlockSize;
                    int srcOffset = rep * currentBlockSize;

                    // Copy the first block
                    Array.Copy(this.Data, srcOffset, result.Data, destOffset, currentBlockSize);

                    // Tile this block
                    for (int m = 1; m < multiples[dim]; m++)
                    {
                        Array.Copy(result.Data, destOffset, result.Data, destOffset + (m * currentBlockSize), currentBlockSize);
                    }
                }

                currentRepetitions *= multiples[dim];
                currentBlockSize = nextBlockSize;
            }

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

            // Create a padded version of the original shape
            int[] paddedShape = new int[newShape.Length];
            for (int i = 0; i < paddedShape.Length; i++)
            {
                paddedShape[i] = 1;
            }

            // Copy the original shape to the rightmost positions
            int offset = newShape.Length - this.Shape.Length;
            Array.Copy(this.Shape, 0, paddedShape, offset, this.Shape.Length);

            // Ensure the shape is compatible for broadcasting
            for (int i = 0; i < newShape.Length; i++)
            {
                if (paddedShape[i] != 1 && paddedShape[i] != newShape[i])
                {
                    // Check if target dimension is divisible by the original dimension
                    if (newShape[i] % paddedShape[i] != 0)
                    {
                        throw new ArgumentException($"Shape mismatch at dimension {i}. Original size {paddedShape[i]} cannot be broadcast to {newShape[i]}.");
                    }
                }
            }

            // Calculate the total size of the new shape
            int newTotalSize = newShape.Aggregate(1, (a, b) => a * b);
            var broadcastedData = PradTools.AllocateArray(newTotalSize);

            // Fill the new data array by repeating the elements
            Parallel.For(0, newTotalSize, i =>
            {
                broadcastedData[i] = this.Data[this.GetOldIndex(i, newShape, paddedShape)];
            });

            return new Tensor(newShape, broadcastedData);
        }

        /// <summary>
        /// Computes the corresponding index in the original tensor's data for the given broadcasted index.
        /// </summary>
        /// <param name="broadcastedIndex">Index in the broadcasted data array.</param>
        /// <param name="newShape">The shape of the broadcasted tensor.</param>
        /// <param name="paddedShape">The padded shape of the original tensor.</param>
        /// <returns>The corresponding index in the original tensor's data array.</returns>
        public int GetOldIndex(int broadcastedIndex, int[] newShape, int[] paddedShape)
        {
            // First, convert the flat index to coordinates in the new shape
            int[] newCoords = new int[newShape.Length];
            int remainingI = broadcastedIndex;
            for (int j = newShape.Length - 1; j >= 0; j--)
            {
                newCoords[j] = remainingI % newShape[j];
                remainingI /= newShape[j];
            }

            // Calculate the corresponding index in the original tensor
            int oldIndex = 0;
            int stride = 1;

            // Start from the rightmost dimension (which is aligned with the original shape)
            for (int j = newShape.Length - 1; j >= 0; j--)
            {
                // For dimensions in the original tensor
                if (paddedShape[j] > 1)
                {
                    // If this dimension is repeated (broadcast from smaller to larger)
                    if (newShape[j] > paddedShape[j] && newShape[j] % paddedShape[j] == 0)
                    {
                        // Use modulo to wrap around for repeated blocks
                        oldIndex += (newCoords[j] % paddedShape[j]) * stride;
                    }
                    else
                    {
                        oldIndex += newCoords[j] * stride;
                    }

                    stride *= paddedShape[j];
                }

                // For dimensions added through padding (original dim was 1)
                // No need to add to oldIndex since these don't contribute to the index
            }

            return oldIndex;
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
        /// Computes the reverse gradient for the slice operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="begin">The starting indices for each axis.</param>
        /// <param name="size">The lengths of the slice along each axis.</param>
        /// <param name="strides">The step size for each axis (default is 1).</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor SliceReverse(Tensor upstreamGradient, int[] begin, int[] size, int[]? strides = null)
        {
            Tensor inputTensor = this; // Assuming this tensor is the original input

            if (begin.Length != inputTensor.Shape.Length || size.Length != inputTensor.Shape.Length || (strides != null && strides.Length != inputTensor.Shape.Length))
            {
                throw new ArgumentException("The lengths of begin, size, and strides must match the number of dimensions of the tensor.");
            }

            strides ??= Enumerable.Repeat(1, inputTensor.Shape.Length).ToArray();
            if (strides.Any(s => s == 0))
            {
                throw new ArgumentException("Stride cannot be zero.");
            }

            int[] adjustedBegin = AdjustIndicesForNegativeValues(begin, inputTensor.Shape);
            int[] effectiveSize = CalculateEffectiveSize(size, adjustedBegin, strides, inputTensor.Shape);

            var inputGradient = new Tensor(inputTensor.Shape);

            Parallel.For(0, upstreamGradient.Data.Length, upstreamIndex =>
            {
                int[] resultIndices = upstreamGradient.GetMultiDimensionalIndices(upstreamIndex, upstreamGradient.Shape);
                int[] sourceIndices = new int[resultIndices.Length];
                for (int i = 0; i < resultIndices.Length; i++)
                {
                    sourceIndices[i] = adjustedBegin[i] + (resultIndices[i] * strides[i]);
                }

                if (IsWithinBounds(sourceIndices, inputTensor.Shape))
                {
                    inputGradient[sourceIndices] += upstreamGradient[resultIndices];
                }
            });

            return inputGradient;
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

            if (this.Shape.Length == 2 && permutation.SequenceEqual(new int[] { 1, 0 }))
            {
                // 2D case
                int rows = this.Shape[0];
                int cols = this.Shape[1];

                Blas.omatcopy(LayoutChar.RowMajor, TransChar.Yes, rows, cols, PradTools.One, this.Data, cols, result.Data, rows);
            }
            else if (this.Shape.Length == 3 && permutation.SequenceEqual(new int[] { 0, 2, 1 }))
            {
                // 3D batch case
                int batchSize = this.Shape[0];
                int rows = this.Shape[1];
                int cols = this.Shape[2];
                int sliceSize = rows * cols;
                var resultSlice = PradTools.AllocateArray(sliceSize);

                // We'll treat this as a batch of 2D matrices
                for (int b = 0; b < batchSize; b++)
                {
                    var slice = this.Data.AsSpan(b * sliceSize, sliceSize);

                    Blas.omatcopy(
                        LayoutChar.RowMajor,
                        TransChar.Yes,
                        rows,
                        cols,
                        PradTools.One,
                        slice.ToArray(),
                        cols,
                        resultSlice,
                        rows);

                    Buffer.BlockCopy(resultSlice, 0, result.Data, b * sliceSize * PradTools.SizeOf, sliceSize * PradTools.SizeOf);
                }
            }
            else
            {
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
            }

            return result;
        }

        /// <summary>
        /// An indexer for the tensor that supports slicing with null values as indices.
        /// </summary>
        /// <param name="indices">The indices used to slice. Null values select the entire dimension.</param>
        /// <returns>The sliced tensor.</returns>
        /// <exception cref="ArgumentException">Number of indices does not match rank.</exception>
        public Tensor Indexer(params string?[] indices)
        {
            // Handle '...' (ellipsis) by expanding it to select all preceding dimensions
            indices = this.ExpandEllipsis(indices, this.Shape.Length);

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
        /// Extracts patches from a 2D matrix tensor (using MKLNET for optimization).
        /// </summary>
        /// <param name="filterSize">The size of the sliding window [filter_height, filter_width].</param>
        /// <param name="strides">The strides for the sliding window [stride_height, stride_width].</param>
        /// <param name="padding">Padding type ('VALID' or 'SAME').</param>
        /// <returns>A new tensor containing the extracted patches.</returns>
        public Tensor ExtractPatches(int[] filterSize, int[] strides, string padding)
        {
            if (filterSize.Length != 2 || strides.Length != 2)
            {
                throw new ArgumentException("Filter size and strides must have 2 dimensions (height, width).");
            }

            // Determine if the tensor is 2D, 3D, or already has a batch dimension (4D)
            Tensor input;
            int batchSize;
            int channels;

            if (this.Shape.Length == 2)
            {
                // 2D tensor: expand to [1, height, width, 1] (add both batch and channel dimensions)
                input = this.ExpandDims(0).ExpandDims(-1);  // Expand to [1, height, width, 1]
                batchSize = 1;
                channels = 1;                               // Single channel
            }
            else if (this.Shape.Length == 3)
            {
                // 3D tensor: expand to [1, height, width, channels] (add batch dimension only)
                input = this.ExpandDims(0);                 // Expand to [1, height, width, channels]
                batchSize = 1;
                channels = this.Shape[2];                   // Multiple channels
            }
            else if (this.Shape.Length == 4)
            {
                // 4D tensor: already has batch and channel dimensions [batch, height, width, channels]
                input = this;                               // No expansion needed
                batchSize = this.Shape[0];                  // Batch size
                channels = this.Shape[3];                   // Number of channels
            }
            else
            {
                throw new ArgumentException("Input tensor must be 2D, 3D, or 4D.");
            }

            int inputHeight = input.Shape[1];  // [batch, height, width, channels]
            int inputWidth = input.Shape[2];   // [batch, height, width, channels]
            int filterHeight = filterSize[0];
            int filterWidth = filterSize[1];
            int strideHeight = strides[0];
            int strideWidth = strides[1];

            int padTop, padBottom, padLeft, padRight;

            if (padding == "SAME")
            {
                padTop = (filterHeight - 1) / 2;
                padBottom = (filterHeight - 1) - padTop;
                padLeft = (filterWidth - 1) / 2;
                padRight = (filterWidth - 1) - padLeft;
            }
            else if (padding == "VALID")
            {
                padTop = padBottom = padLeft = padRight = 0;  // No padding for 'VALID'
            }
            else
            {
                throw new ArgumentException("Unsupported padding type. Use 'VALID' or 'SAME'.");
            }

            // Apply padding to the input tensor if required
            Tensor paddedInput = input.PadUsingMKL(padTop, padBottom, padLeft, padRight);

            // Calculate output dimensions
            int outHeight = ((paddedInput.Shape[1] - filterHeight) / strideHeight) + 1;
            int outWidth = ((paddedInput.Shape[2] - filterWidth) / strideWidth) + 1;

            // Initialize the output tensor to hold the patches
            int patchSize = filterHeight * filterWidth * channels;
            var outputShape = new int[] { batchSize, outHeight, outWidth, patchSize };
            var size = batchSize * outHeight * outWidth * patchSize;
            var outputData = PradTools.AllocateArray(size);

            // Precompute channel stride for efficiency
            int channelStride = paddedInput.Shape[1] * paddedInput.Shape[2] * channels;

            Parallel.For(0, batchSize, b =>
            {
                int batchOffset = b * channelStride;

                for (int i = 0; i < outHeight; i++)
                {
                    for (int j = 0; j < outWidth; j++)
                    {
                        int patchIndex = ((b * outHeight * outWidth) + (i * outWidth) + j) * patchSize;

                        for (int h = 0; h < filterHeight; h++)
                        {
                            int srcY = (i * strideHeight) + h;
                            int srcXStart = j * strideWidth;
                            int dstIndex = patchIndex + (h * filterWidth * channels);

                            // Copy the entire row for all channels at once
                            int srcOffset = batchOffset + (((srcY * paddedInput.Shape[2]) + srcXStart) * channels);
                            Buffer.BlockCopy(
                                paddedInput.Data,
                                srcOffset * PradTools.SizeOf,
                                outputData,
                                dstIndex * PradTools.SizeOf,
                                filterWidth * channels * PradTools.SizeOf);
                        }
                    }
                }
            });

            return new Tensor(outputShape, outputData);
        }

        /// <summary>
        /// Pads a tensor using MKLNET operations.
        /// </summary>
        /// <param name="padTop">Padding for the top.</param>
        /// <param name="padBottom">Padding for the bottom.</param>
        /// <param name="padLeft">Padding for the left.</param>
        /// <param name="padRight">Padding for the right.</param>
        /// <returns>A new padded tensor.</returns>
        public Tensor PadUsingMKL(int padTop, int padBottom, int padLeft, int padRight)
        {
            int batchSize, height, width, channels;

            if (this.Shape.Length == 2)
            {
                batchSize = 1;
                height = this.Shape[0];
                width = this.Shape[1];
                channels = 1;
            }
            else if (this.Shape.Length == 3)
            {
                batchSize = 1;
                height = this.Shape[0];
                width = this.Shape[1];
                channels = this.Shape[2];
            }
            else if (this.Shape.Length == 4)
            {
                batchSize = this.Shape[0];
                height = this.Shape[1];
                width = this.Shape[2];
                channels = this.Shape[3];
            }
            else
            {
                throw new ArgumentException("Input tensor must be 2D, 3D, or 4D.");
            }

            int paddedHeight = height + padTop + padBottom;
            int paddedWidth = width + padLeft + padRight;

            var outputShape = this.Shape.Length == 2
                ? new int[] { paddedHeight, paddedWidth }
                : new int[] { batchSize, paddedHeight, paddedWidth, channels };

            var size = batchSize * paddedHeight * paddedWidth * channels;
            var paddedData = PradTools.AllocateArray(size);

            int inputStride = width * channels;
            int outputStride = paddedWidth * channels;

            Parallel.For(0, batchSize, b =>
            {
                int inputBatchOffset = b * height * inputStride;
                int outputBatchOffset = b * paddedHeight * outputStride;

                for (int i = 0; i < height; i++)
                {
                    int inputRowOffset = inputBatchOffset + (i * inputStride);
                    int outputRowOffset = outputBatchOffset + ((i + padTop) * outputStride) + (padLeft * channels);

                    Buffer.BlockCopy(
                        this.Data,
                        inputRowOffset * PradTools.SizeOf,
                        paddedData,
                        outputRowOffset * PradTools.SizeOf,
                        inputStride * PradTools.SizeOf);
                }
            });

            return new Tensor(outputShape, paddedData);
        }

        /// <summary>
        /// Performs element-wise less than comparison with another tensor or scalar.
        /// </summary>
        /// <param name="other">The tensor or scalar to compare with.</param>
        /// <returns>A new tensor containing the boolean mask.</returns>
        public Tensor LessThan(Tensor other)
        {
            if (!this.Shape.SequenceEqual(other.Shape))
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise comparison.");
            }

            var result = new Tensor(this.Shape);
            int vectorSize = PradTools.VectorCount();

            for (int i = 0; i <= this.Data.Length - vectorSize; i += vectorSize)
            {
                var thisVector = PradTools.AllocateVector(this.Data, i);
                var otherVector = PradTools.AllocateVector(other.Data, i);
                var comparisonVector = Vector.LessThan(thisVector, otherVector);
                var mask = PradTools.Convert(Vector.Abs(comparisonVector));
                mask.CopyTo(result.Data, i);
            }

            // Handle remaining elements
            for (int i = this.Data.Length - (this.Data.Length % vectorSize); i < this.Data.Length; i++)
            {
                result.Data[i] = this.Data[i] < other.Data[i] ? PradTools.One : PradTools.Zero;
            }

            return result;
        }

        /// <summary>
        /// Performs element-wise greater than comparison with another tensor or scalar.
        /// </summary>
        /// <param name="other">The tensor or scalar to compare with.</param>
        /// <returns>A new tensor containing the boolean mask.</returns>
        public Tensor GreaterThan(Tensor other)
        {
            if (!this.Shape.SequenceEqual(other.Shape))
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise comparison.");
            }

            var result = new Tensor(this.Shape);
            int vectorSize = PradTools.VectorCount();

            for (int i = 0; i <= this.Data.Length - vectorSize; i += vectorSize)
            {
                var thisVector = PradTools.AllocateVector(this.Data, i);
                var otherVector = PradTools.AllocateVector(other.Data, i);
                var comparisonVector = Vector.GreaterThan(thisVector, otherVector);
                var mask = PradTools.Convert(Vector.Abs(comparisonVector));
                mask.CopyTo(result.Data, i);
            }

            // Handle remaining elements
            for (int i = this.Data.Length - (this.Data.Length % vectorSize); i < this.Data.Length; i++)
            {
                result.Data[i] = this.Data[i] > other.Data[i] ? PradTools.One : PradTools.Zero;
            }

            return result;
        }

        /// <summary>
        /// Performs element-wise modulus operation with another tensor.
        /// </summary>
        /// <param name="other">The tensor to perform modulus with.</param>
        /// <returns>A new tensor containing the element-wise modulus results.</returns>
        public Tensor Modulus(Tensor other)
        {
            if (!this.Shape.SequenceEqual(other.Shape))
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise operations.");
            }

            var result = new Tensor(this.Shape);

            // Allocate temporary arrays for integral and fractional parts
            var integral = PradTools.AllocateArray(this.Data.Length);
            var fractional = PradTools.AllocateArray(this.Data.Length);

            // Use modf to separate the fractional and integral parts of the division
            Vml.Div(this.Data, other.Data, result.Data);    // result.Data = this.Data / other.Data
            Vml.Modf(result.Data, integral, fractional);    // integral = truncated int part, fractional = remaining fraction

            // Compute element-wise modulus using: this.Data - (integral * other.Data)
            Vml.Mul(integral, other.Data, integral);        // integral = integral * other.Data
            Vml.Sub(this.Data, integral, result.Data);      // result.Data = this.Data - integral

            return result;
        }

        /// <summary>
        /// Computes the softmax of the tensor along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to apply the softmax on (default is last axis).</param>
        /// <returns>A tensor with softmax values.</returns>
        public Tensor Softmax(int axis = -1)
        {
            int[] shape = this.Shape;
            var result = new Tensor(shape);

            if (axis < 0)
            {
                axis = shape.Length + axis;
            }

            int outerSize = shape.Take(axis).Aggregate(1, (x, y) => x * y);
            int axisSize = shape[axis];
            int innerSize = shape.Skip(axis + 1).Aggregate(1, (x, y) => x * y);

            Parallel.For(0, outerSize, i =>
            {
                for (int k = 0; k < innerSize; k++)
                {
                    // Find start index of current row
                    int startIndex = (i * axisSize * innerSize) + k;

                    // Compute max value for numerical stability
                    var maxVal = this.Data[startIndex];
                    for (int j = 1; j < axisSize; j++)
                    {
                        maxVal = Math.Max(maxVal, this.Data[startIndex + (j * innerSize)]);
                    }

                    // Calculate exp(logits - maxVal)
                    var sum = PradTools.Zero;
                    for (int j = 0; j < axisSize; j++)
                    {
                        result.Data[startIndex + (j * innerSize)] =
                            PradMath.Exp(this.Data[startIndex + (j * innerSize)] - maxVal);
                        sum += result.Data[startIndex + (j * innerSize)];
                    }

                    // Normalize by dividing by the sum
                    for (int j = 0; j < axisSize; j++)
                    {
                        result.Data[startIndex + (j * innerSize)] /= sum;
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Computes the mean and variance of the tensor along the specified axes.
        /// </summary>
        /// <param name="axes">The axes along which to compute mean and variance. If null, reduce across all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns>A tuple (mean, variance) of tensors.</returns>
        public (Tensor mean, Tensor variance) Moments(int[]? axes = null, bool keepDims = false)
        {
            int[] shape = this.Shape;
            int rank = shape.Length;

            if (axes == null)
            {
                axes = Enumerable.Range(0, rank).ToArray();  // Reduce across all dimensions if axes are not specified
            }

            // Ensure axes are positive and sorted
            axes = axes.Select(axis => axis < 0 ? rank + axis : axis).OrderBy(x => x).ToArray();

            // Calculate the shape of the reduced tensor
            int[] reducedShape = shape.ToArray();
            foreach (var axis in axes)
            {
                reducedShape[axis] = 1;
            }

            // Compute mean
            var mean = this.ReduceSum(axes).Divide(this.Data.Length / reducedShape.Aggregate(1, (a, b) => a * b));

            // Compute variance using Welford's algorithm for numerical stability
            var variance = new Tensor(mean.Shape);

            Parallel.For(0, this.Data.Length, i =>
            {
                int[] indices = this.GetMultiDimensionalIndices(i, shape);
                int reducedIndex = this.MapToReducedIndex(indices, axes, reducedShape);

                var delta = this.Data[i] - mean.Data[reducedIndex];
                variance.Data[reducedIndex] += delta * delta;
            });

            // Normalize variance by N
            variance = variance.Divide(this.Data.Length / reducedShape.Aggregate(1, (a, b) => a * b));

            // Squeeze reduced dimensions if keepDims is false
            if (!keepDims)
            {
                mean = mean.Squeeze(axes);
                variance = variance.Squeeze(axes);
            }

            return (mean, variance);
        }

        /// <summary>
        /// Finds the indices of the maximum values along the specified axis with optimized memory access and vectorized operations using Vector of double.
        /// </summary>
        /// <param name="axis">The axis along which to find the maximum indices. If -1, finds the index of the maximum value in the flattened tensor.</param>
        /// <returns>A new tensor containing the indices of the maximum values.</returns>
        public Tensor ArgMax(int axis = -1)
        {
            // Handle negative axis
            if (axis < 0)
            {
                axis = this.Shape.Length + axis;
            }

            // Validate the axis
            if (axis < 0 || axis >= this.Shape.Length)
            {
                throw new ArgumentException($"Axis value {axis} is out of bounds for tensor with {this.Shape.Length} dimensions.");
            }

            int[] outputShape = this.Shape.Take(axis).Concat(this.Shape.Skip(axis + 1)).ToArray();
            var result = new Tensor(outputShape); // Create a new tensor to hold the indices of the maximum values

            int axisLength = this.Shape[axis]; // The size of the dimension along which we're finding ArgMax
            int outerSize = this.Shape.Take(axis).Aggregate(1, (x, y) => x * y); // Total size before the axis
            int innerSize = this.Shape.Skip(axis + 1).Aggregate(1, (x, y) => x * y); // Total size after the axis

            // Precompute strides to optimize index computation
            int[] strides = ComputeStrides(this.Shape);

            // Vectorized width (Vector<double> typically operates on 2 doubles at once)
            int vectorWidth = Vector<double>.Count;

            // Parallel loop to compute ArgMax for each slice along the specified axis
            Parallel.For(0, outerSize, i =>
            {
                for (int j = 0; j < innerSize; j++)
                {
                    int startIndex = (i * axisLength * innerSize) + j; // Starting index of the slice
                    var maxValue = this.Data[startIndex];
                    int maxIndex = 0;

                    // Vectorized comparison (SIMD)
                    int k = 1;
                    for (; k + vectorWidth <= axisLength; k += vectorWidth)
                    {
                        int currentIndex = startIndex + (k * innerSize);

                        // Load the current values into vectors
                        var currentVector = PradTools.AllocateVector(this.Data, currentIndex);
                        var maxVector = PradTools.AllocateVector(maxValue);

                        // Perform element-wise max comparison
                        var comparison = Vector.Max(currentVector, maxVector);

                        // Check which values are greater and update maxValue and maxIndex
                        for (int v = 0; v < vectorWidth; v++)
                        {
                            if (comparison[v] > maxValue)
                            {
                                maxValue = comparison[v];
                                maxIndex = k + v;
                            }
                        }
                    }

                    // Handle remaining elements that don't fit into the vector size
                    for (; k < axisLength; k++)
                    {
                        int currentIndex = startIndex + (k * innerSize);
                        if (this.Data[currentIndex] > maxValue)
                        {
                            maxValue = this.Data[currentIndex];
                            maxIndex = k;
                        }
                    }

                    // Store the index of the maximum value in the result tensor
                    result.Data[(i * innerSize) + j] = maxIndex;
                }
            });

            return result;
        }

        /// <summary>
        /// Performs an element-wise floor operation on the tensor.
        /// </summary>
        /// <returns>A new tensor containing the element-wise floor results.</returns>
        public Tensor ElementwiseFloor()
        {
            var result = new Tensor(this.Shape);

            // Apply floor to each element in the tensor
            Vml.Floor(this.Data, result.Data);

            return result;
        }

        /// <summary>
        /// Performs an element-wise negation of the tensor.
        /// </summary>
        /// <returns>A new tensor containing the element-wise negated results.</returns>
        public Tensor ElementwiseNegate()
        {
            var result = new Tensor(this.Shape);

            // Multiply each element by -1 to negate it
            var temp = PradTools.AllocateArray(this.Data.Length);
            Array.Fill(temp, PradTools.NegativeOne);
            Vml.Mul(this.Data, temp, result.Data);

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
        /// Computes the element-wise power of the tensor to the given exponent using MKL.NET.
        /// </summary>
        /// <param name="exponent">The exponent to raise each element to. Can be a scalar double or a Tensor.</param>
        /// <returns>A new tensor with the result of the power operation.</returns>
        /// <exception cref="ArgumentException">Thrown when the exponent tensor shape doesn't match or the exponent type is invalid.</exception>
        public Tensor Pow(object exponent)
        {
            if (exponent is float scalarExponentF)
            {
                return this.PowHelper(scalarExponentF);
            }
            else if (exponent is double scalarExponent)
            {
                return this.PowHelper(scalarExponent);
            }
            else if (exponent is Tensor exponentTensor)
            {
                if (!this.Shape.SequenceEqual(exponentTensor.Shape))
                {
                    throw new ArgumentException("The shapes of the tensors must match for element-wise power operation.");
                }

                var result = new Tensor(this.Shape);
                Vml.Pow(this.Data.Length, this.Data, exponentTensor.Data, result.Data);
                return result;
            }
            else
            {
                throw new ArgumentException("Exponent must be either a float, double, or a Tensor.");
            }
        }

        /// <summary>
        /// Computes the entropy of the tensor along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to compute the entropy. If null, compute entropy over the entire tensor.</param>
        /// <returns>A tensor containing the entropy along the specified axis.</returns>
        public Tensor Entropy(int? axis = null)
        {
            // Validate that the tensor represents a probability distribution (values between 0 and 1).
            this.ValidateProbabilities();

            // Apply log operation to the tensor (element-wise log)
            var logTensor = this.Log();

            // Multiply the probabilities by their log values (element-wise multiplication)
            var entropyTensor = this.ElementwiseMultiply(logTensor);

            // Sum the results along the specified axis and negate to compute entropy
            var summedEntropy = axis.HasValue ? entropyTensor.Sum(new int[] { axis.Value }) : entropyTensor.Sum(Enumerable.Range(0, entropyTensor.Shape.Length).ToArray());

            return summedEntropy.ElementwiseNegate(); // Return -sum(P(x) * log(P(x)))
        }

        /// <summary>
        /// Generates a tensor filled with uniformly distributed random values in the range [minValue, maxValue].
        /// </summary>
        /// <param name="minValue">The minimum value of the uniform distribution (inclusive).</param>
        /// <param name="maxValue">The maximum value of the uniform distribution (exclusive).</param>
        /// <returns>A tensor filled with uniformly distributed random values.</returns>
        public Tensor Uniform(double minValue = 0.0, double maxValue = 1.0)
        {
            var shape = this.Shape;

            // Validate the input range
            if (minValue >= maxValue)
            {
                throw new ArgumentException("minValue must be less than maxValue.");
            }

            // Create the result tensor with the specified shape
            var result = new Tensor(shape);
            var totalSize = result.Data.Length;
            double range = maxValue - minValue;

            // Use Parallel.For for thread-safe random number generation using ThreadLocal<Random>
            Parallel.For(0, totalSize, i =>
            {
                // Use ThreadLocal Random to avoid conflicts
                result.Data[i] = PradTools.Cast((RandomGen.Value.NextDouble() * range) + minValue);
            });

            return result;
        }

        /// <summary>
        /// Computes the element-wise modified Bessel function of the first kind, I0(x), of the tensor's elements.
        /// </summary>
        /// <returns>A new tensor containing the result of I0(x) for each element.</returns>
        public Tensor BesselI0()
        {
            // Create a result tensor with the same shape as the current tensor
            var result = new Tensor(this.Shape);

            // Precompute constants for small and large value approximations
            const double threshold = 3.75;
            const double invThreshold = 1 / threshold; // Save division later
            const double sqrt2pi = 0.39894228; // Approximation constant for large values

            // Constants for small value approximation
            double[] smallCoeffs = { 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813 };

            // Constants for large value approximation
            double[] largeCoeffs = { 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537, -0.01647633, 0.00392377 };

            for (int i = 0; i < this.Data.Length; i++)
            {
                double absX = Math.Abs(this.Data[i]);

                // Apply the small value approximation for abs(x) < 3.75
                if (absX < threshold)
                {
                    double t = this.Data[i] * invThreshold;
                    double t2 = t * t;

                    // Horner's method for polynomial evaluation
                    result.Data[i] = PradTools.Cast(1 + (t2 * (smallCoeffs[0] + (t2 * (smallCoeffs[1] + (t2 * (smallCoeffs[2] +
                                   (t2 * (smallCoeffs[3] + (t2 * (smallCoeffs[4] + (t2 * smallCoeffs[5]))))))))))));
                }

                // Apply the large value approximation for abs(x) >= 3.75
                else
                {
                    double t = threshold / absX;
                    result.Data[i] = PradTools.Cast((Math.Exp(absX) / Math.Sqrt(absX)) *
                                     (sqrt2pi + (t * (largeCoeffs[0] + (t * (largeCoeffs[1] + (t * (largeCoeffs[2] +
                                     (t * (largeCoeffs[3] + (t * (largeCoeffs[4] + (t * (largeCoeffs[5] +
                                     (t * (largeCoeffs[6] + (t * largeCoeffs[7])))))))))))))))));
                }
            }

            return result;
        }

        /// <summary>
        /// Upsamples the tensor using the specified scaling factor and interpolation method.
        /// Supports 2D and 4D tensors (batch, height, width, channels).
        /// </summary>
        /// <param name="scaleFactor">The scaling factor for both height and width.</param>
        /// <param name="method">The interpolation method: "nearest" (default) or "bilinear".</param>
        /// <returns>A new tensor that has been upsampled by the given factor.</returns>
        public Tensor Upsample(int scaleFactor, string method = "nearest")
        {
            if (scaleFactor <= 0)
            {
                throw new ArgumentException("Scale factor must be greater than 0.");
            }

            // Determine the dimensionality and shape of the input tensor
            int[] inputShape = this.Shape;
            Tensor input = this;

            int batchSize, channels, inputHeight, inputWidth;

            if (inputShape.Length == 2)
            {
                // Assume it's a 2D tensor, treat it as a single-channel image
                batchSize = 1;
                channels = 1;
                inputHeight = inputShape[0];
                inputWidth = inputShape[1];
            }
            else if (inputShape.Length == 3)
            {
                // Single image with channels [height, width, channels]
                batchSize = 1;
                inputHeight = inputShape[0];
                inputWidth = inputShape[1];
                channels = inputShape[2];
            }
            else if (inputShape.Length == 4)
            {
                // Batch of images [batch, height, width, channels]
                batchSize = inputShape[0];
                inputHeight = inputShape[1];
                inputWidth = inputShape[2];
                channels = inputShape[3];
            }
            else
            {
                throw new ArgumentException("Upsampling only supports 2D or 4D tensors.");
            }

            // Calculate the new output dimensions
            int outputHeight = inputHeight * scaleFactor;
            int outputWidth = inputWidth * scaleFactor;

            // Create the upsampled tensor
            var outputShape = new int[] { batchSize, outputHeight, outputWidth, channels };
            var result = new Tensor(outputShape);

            // Nearest neighbor upsampling
            if (method == "nearest")
            {
                Parallel.For(0, batchSize, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            int nearestY = i / scaleFactor;
                            for (int j = 0; j < outputWidth; j++)
                            {
                                int nearestX = j / scaleFactor;

                                // Copy the value from the nearest pixel in the input tensor
                                result[b, i, j, c] = input[b, nearestY, nearestX, c];
                            }
                        }
                    }
                });
            }

            // Bilinear interpolation upsampling
            else if (method == "bilinear")
            {
                Parallel.For(0, batchSize, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            float srcY = (float)i / scaleFactor;
                            int y0 = (int)Math.Floor(srcY);
                            int y1 = Math.Min(y0 + 1, inputHeight - 1);
                            float dy = srcY - y0;

                            for (int j = 0; j < outputWidth; j++)
                            {
                                var srcX = (float)j / scaleFactor;
                                int x0 = (int)Math.Floor(srcX);
                                int x1 = Math.Min(x0 + 1, inputWidth - 1);
                                var dx = srcX - x0;

                                // Bilinear interpolation: top-left, top-right, bottom-left, bottom-right contributions
                                var top = ((1 - dx) * input[b, y0, x0, c]) + (dx * input[b, y0, x1, c]);
                                var bottom = ((1 - dx) * input[b, y1, x0, c]) + (dx * input[b, y1, x1, c]);

                                result[b, i, j, c] = PradTools.Cast(((1 - dy) * top) + (dy * bottom));
                            }
                        }
                    }
                });
            }
            else
            {
                throw new ArgumentException($"Unsupported method: {method}. Use 'nearest' or 'bilinear'.");
            }

            return result;
        }

        /// <summary>
        /// Performs an interleaved gather operation on the tensor with efficient copying and correct looping logic.
        /// </summary>
        /// <param name="skip">The number of row indices to skip during the interleaving process.</param>
        /// <param name="restart">The number of rows before restarting the pattern.</param>
        /// <returns>A new tensor with interleaved gathered results.</returns>
        public Tensor InterleavedGather(int skip, int restart)
        {
            // Step 1: Reshape the tensor to [3, 18] (assuming original shape [3, 3, 6])
            int[] reshapedShape = new int[] { this.Shape[0], this.Shape[1] * this.Shape[2] };
            Tensor reshapedTensor = this.Reshape(reshapedShape);

            // Step 2: Transpose the reshaped tensor to [18, 3] for interleaving
            Tensor transposedTensor = reshapedTensor.Transpose(1, 0); // Transpose to [18, 3]

            // Step 3: Prepare result tensor with shape [number of interleaved rows, batch size (3)]
            int totalRowsToGather = transposedTensor.Shape[0];  // You are gathering all rows from the transposed tensor
            Tensor resultTensor = new Tensor(new int[] { totalRowsToGather, transposedTensor.Shape[1] }); // Shape [18, 3]

            int resultIndex = 0;  // Index for storing values in the result tensor

            // Single loop for i++
            for (int i = 0; i < transposedTensor.Shape[0]; i++)
            {
                // Copy the row at index i
                Array.Copy(transposedTensor.Data, i * this.Shape[0], resultTensor.Data, resultIndex * this.Shape[0], this.Shape[0]);
                resultIndex++;

                Array.Copy(transposedTensor.Data, (i + skip) * this.Shape[0], resultTensor.Data, resultIndex * this.Shape[0], this.Shape[0]);
                resultIndex++;

                // Only reset i when we have completed the restart block
                if (resultIndex % restart == 0)
                {
                    i += skip;  // Move i to the next block for the next iteration
                }
            }

            // Step 4: Reshape the result back to the original shape (e.g., [1, 3, 3, 6])
            return resultTensor.Reshape(new int[] { 1, this.Shape[1], skip, this.Shape[0] * 2 });
        }

        /// <summary>
        /// Performs the inverse of an interleaved gather operation on the tensor, restoring it to its original structure.
        /// </summary>
        /// <param name="skip">The number of row indices that were skipped during the interleaving process.</param>
        /// <param name="restart">The number of rows before the restart pattern occurred during interleaving.</param>
        /// <returns>A tensor that has been restored to its original structure.</returns>
        public Tensor InterleavedGatherInverse(int skip, int restart)
        {
            // Step 1: Reshape the tensor from [1, 3, 3, 6] to [9, 6] (flatten matrix)
            int totalRowsToGather = (this.Shape[^1] / 2) * this.Shape[1];
            int[] reshapedShape = new int[] { this.Shape[1] * this.Shape[2], this.Shape[^1] }; // [9, 6]
            Tensor reshapedTensor = this.Reshape(reshapedShape);

            // Step 2: Transpose from [9, 6] to [6, 9] to separate components
            Tensor transposedTensor = reshapedTensor.Transpose(1, 0); // [6, 9]

            // Step 3: Prepare the result tensor, where we will gather the interleaved rows
            Tensor resultTensor = new Tensor(reshapedShape); // [9, 6]

            int resultIndex = 0;

            // Step 4: Perform 6 array copies per iteration to undo the interleaving
            for (int iteration = 0; iteration < this.Shape[1]; iteration++)
            {
                // Step 4.1: Copy magnitudes for the current channel (e.g., `mr` for red)
                Array.Copy(transposedTensor.Data, iteration * totalRowsToGather, resultTensor.Data, resultIndex * restart, skip);    // First set of magnitudes
                Array.Copy(transposedTensor.Data, (iteration * totalRowsToGather) + skip, resultTensor.Data, (resultIndex + 1) * restart, skip); // Second set
                Array.Copy(transposedTensor.Data, (iteration * totalRowsToGather) + (skip * 2), resultTensor.Data, (resultIndex + 2) * restart, skip); // Third set

                // Step 4.2: Copy angles for the current channel (e.g., `ar` for red)
                Array.Copy(transposedTensor.Data, (iteration + skip) * totalRowsToGather, resultTensor.Data, (resultIndex * restart) + skip, skip);  // First set of angles
                Array.Copy(transposedTensor.Data, ((iteration + skip) * totalRowsToGather) + skip, resultTensor.Data, ((resultIndex + 1) * restart) + skip, skip); // Second set
                Array.Copy(transposedTensor.Data, ((iteration + skip) * totalRowsToGather) + (skip * 2), resultTensor.Data, ((resultIndex + 2) * restart) + skip, skip); // Third set

                // Step 4.3: Move to the next block of rows for the next iteration (g, b channels)
                resultIndex += 3;
            }

            // Step 5: Reshape the result back to [3, 3, 6] (original shape)
            return resultTensor.Reshape(new int[] { (this.Shape[^1] / 2), this.Shape[1], this.Shape[^2] * 2 });
        }

        /// <summary>
        /// Computes the Mean Squared Error (MSE) between two tensors, with an option to treat the first dimension as a batch dimension.
        /// </summary>
        /// <param name="yPred">The tensor of predicted values.</param>
        /// <param name="hasBatchDimension">If true, treats the first dimension as the batch dimension and computes MSE per batch. If false, computes MSE over the entire tensor.</param>
        /// <returns>A tensor containing the MSE for each batch if hasBatchDimension is true, otherwise a scalar tensor with the overall MSE.</returns>
        /// <exception cref="ArgumentException">Thrown if the shapes of the input tensors do not match.</exception>
        public (Tensor, Tensor) MeanSquaredError(Tensor yPred, bool hasBatchDimension = false)
        {
            Tensor yTrue = this;

            // Ensure the shapes of the two tensors are the same
            if (!yTrue.Shape.SequenceEqual(yPred.Shape))
            {
                throw new ArgumentException("The tensors must have the same shape for Mean Squared Error calculation.");
            }

            if (hasBatchDimension && yTrue.Shape.Length > 1)
            {
                // Treat the first dimension as the batch dimension
                int batchSize = yTrue.Shape[0];
                int[] reducedShape = yTrue.Shape.Skip(1).ToArray(); // Shape for individual batches

                // Preallocate a tensor to store the MSE for each batch
                var mseBatch = PradTools.AllocateArray(batchSize);

                // Compute MSE for each batch independently
                for (int batch = 0; batch < batchSize; batch++)
                {
                    // Slice out each batch for both yTrue and yPred
                    Tensor yTrueBatch = yTrue.Slice(new int[] { batch, 0 }, reducedShape);
                    Tensor yPredBatch = yPred.Slice(new int[] { batch, 0 }, reducedShape);

                    // Step 1: Compute the element-wise difference (yTrue - yPred)
                    var difference = yTrueBatch.ElementwiseSub(yPredBatch);

                    // Step 2: Square the differences
                    var squaredDifference = difference.ElementwiseSquare();

                    // Step 3: Sum the squared differences and divide by the number of elements
                    var totalSquaredSum = squaredDifference.Data.Sum();
                    int numElements = squaredDifference.Data.Length;

                    // Step 4: Store the MSE for the current batch
                    mseBatch[batch] = totalSquaredSum / numElements;
                }

                var tensorReverse = new TensorReverse(new Tensor[] { yTrue, yPred });
                var upstream = tensorReverse.MeanSquaredErrorReverse(hasBatchDimension);

                // Step 5: Return a tensor containing the MSE for each batch
                return (new Tensor(new int[] { batchSize }, mseBatch), upstream);
            }
            else
            {
                // No batch dimension; compute MSE over the entire tensor
                // Step 1: Compute the element-wise difference (yTrue - yPred)
                var difference = yTrue.ElementwiseSub(yPred);

                // Step 2: Square the differences
                var squaredDifference = difference.ElementwiseSquare();

                // Step 3: Sum the squared differences and divide by the number of elements
                var totalSquaredSum = squaredDifference.Data.Sum();
                int numElements = squaredDifference.Data.Length;

                // Step 4: Return the MSE as a scalar tensor
                var mseValue = totalSquaredSum / numElements;
                var dataArray = PradTools.AllocateArray(1);
                dataArray[0] = mseValue;

                var tensorReverse = new TensorReverse(new Tensor[] { yTrue, yPred });
                var upstream = tensorReverse.MeanSquaredErrorReverse(hasBatchDimension);

                return (new Tensor(new int[] { 1 }, dataArray), upstream);
            }
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
        /// Parse an index, handling null values as selecting the entire dimension.
        /// </summary>
        /// <param name="index">The index, which can be null.</param>
        /// <param name="dimSize">The dimension size.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <param name="step">The step size.</param>
        /// <param name="isSlice">Indicates if the index is a slice.</param>
        /// <exception cref="ArgumentException">Step is zero.</exception>
        internal void ParseIndex(string? index, int dimSize, out int start, out int end, out int step, out bool isSlice)
        {
            if (index == null)
            {
                // If index is null, select the entire dimension
                start = 0;
                end = dimSize;
                step = 1;
                isSlice = true;
                return;
            }

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

        private static int[] AdjustIndicesForNegativeValues(int[] indices, int[] shape)
        {
            return indices.Select((index, i) => index < 0 ? index + shape[i] : index).ToArray();
        }

        private static int[] CalculateEffectiveSize(int[] size, int[] begin, int[] strides, int[] shape)
        {
            return size.Select((s, i) => s < 0 ? (shape[i] - begin[i]) / Math.Abs(strides[i]) : s).ToArray();
        }

        private static bool IsWithinBounds(int[] indices, int[] shape)
        {
            return indices.Zip(shape, (index, dim) => index >= 0 && index < dim).All(isWithin => isWithin);
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

        private static double[] GenerateRandomDoubleArray(int length)
        {
            var random = RandomGen.Value;
            double[] result = new double[length];

            for (int i = 0; i < length; i++)
            {
                result[i] = random.NextDouble();
            }

            return result;
        }

        /// <summary>
        /// Expands '...' (ellipsis) into a full set of indices, selecting all preceding dimensions fully.
        /// </summary>
        /// <param name="indices">The provided indices, which may contain '...'.</param>
        /// <param name="rank">The rank of the tensor (number of dimensions).</param>
        /// <returns>An expanded array of indices where '...' has been replaced by the appropriate number of full-dimension selectors.</returns>
        private string?[] ExpandEllipsis(string?[] indices, int rank)
        {
            // Check if '...' exists in the indices
            int ellipsisIndex = Array.IndexOf(indices, "...");
            if (ellipsisIndex == -1)
            {
                // No ellipsis found, return the indices as-is
                return indices;
            }

            // Replace '...' with null values to select all preceding dimensions fully
            int numMissing = rank - (indices.Length - 1); // Calculate how many dimensions are implied by '...'
            string?[] expandedIndices = new string?[rank];

            // Fill in the indices before the ellipsis
            Array.Copy(indices, 0, expandedIndices, 0, ellipsisIndex);

            // Fill in nulls for the dimensions that '...' represents
            for (int i = ellipsisIndex; i < ellipsisIndex + numMissing; i++)
            {
                expandedIndices[i] = null;  // null means select the entire dimension
            }

            // Copy the remaining indices after the ellipsis
            Array.Copy(indices, ellipsisIndex + 1, expandedIndices, ellipsisIndex + numMissing, indices.Length - ellipsisIndex - 1);

            return expandedIndices;
        }

        /// <summary>
        /// Reduces the tensor by summing along the specified axes.
        /// </summary>
        /// <param name="axes">Axes to reduce.</param>
        /// <returns>The reduced tensor with summed values.</returns>
        private Tensor ReduceSum(int[] axes)
        {
            int[] shape = this.Shape;
            int[] reducedShape = shape.ToArray();

            // Set reduction dimensions to 1
            foreach (var axis in axes)
            {
                reducedShape[axis] = 1;
            }

            var result = new Tensor(reducedShape);

            // Use a lock array to ensure thread safety during accumulation
            object[] locks = new object[result.Data.Length];
            for (int i = 0; i < locks.Length; i++)
            {
                locks[i] = new object();
            }

            Parallel.For(0, this.Data.Length, i =>
            {
                int[] indices = this.GetMultiDimensionalIndices(i, shape);
                int reducedIndex = this.MapToReducedIndex(indices, axes, reducedShape);

                // Accumulate in a thread-safe manner
                lock (locks[reducedIndex])
                {
                    result.Data[reducedIndex] += this.Data[i];
                }
            });

            return result;
        }

        /// <summary>
        /// Maps a tensor's multi-dimensional indices to reduced indices for summation.
        /// </summary>
        /// <param name="indices">Full indices.</param>
        /// <param name="axes">Axes to reduce.</param>
        /// <param name="reducedShape">Shape after reduction.</param>
        /// <returns>The flattened reduced index.</returns>
        private int MapToReducedIndex(int[] indices, int[] axes, int[] reducedShape)
        {
            int[] reducedIndices = indices.ToArray();
            foreach (var axis in axes)
            {
                reducedIndices[axis] = 0;
            }

            return this.GetIndex(reducedIndices, reducedShape);
        }

        /// <summary>
        /// Computes the flattened index from multi-dimensional indices.
        /// </summary>
        /// <param name="indices">The indices to compute from.</param>
        /// <param name="shape">The tensor's shape.</param>
        /// <returns>The flattened index.</returns>
        private int GetIndex(int[] indices, int[] shape)
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

        /// <summary>
        /// Helper method to compute power with a scalar exponent.
        /// </summary>
        /// <param name="exponent">The scalar exponent to raise each element to.</param>
        /// <returns>A new tensor with the result of the power operation.</returns>
        private Tensor PowHelper(double exponent)
        {
            var result = new Tensor(this.Shape);

            if (exponent == 2.0)
            {
                // For squaring, we can use the more efficient Sqr function
                Vml.Sqr(this.Data.Length, this.Data, result.Data);
            }
            else
            {
                // For general exponents, use the Pow function
                var exponentArray = PradTools.FillArray(this.Data.Length, exponent);
                Vml.Pow(this.Data.Length, this.Data, exponentArray, result.Data);
            }

            return result;
        }

        /// <summary>
        /// Computes the strides for a tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>An array representing the strides for each dimension.</returns>
        private int[] PrecomputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
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

        /// <summary>
        /// Validates that the tensor contains valid probability values (all elements should be between 0 and 1).
        /// </summary>
        private void ValidateProbabilities()
        {
            if (this.Data.Any(p => p < 0 || p > 1))
            {
                throw new ArgumentException("Tensor contains values outside the range [0, 1], and is therefore not a valid probability distribution.");
            }
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

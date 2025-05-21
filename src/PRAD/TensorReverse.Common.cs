//------------------------------------------------------------------------------
// <copyright file="TensorReverse.Common.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Threading;
    using System.Threading.Tasks;
    using MKLNET;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Backward functions for tensors.
    /// </summary>
    public partial class TensorReverse
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TensorReverse"/> class.
        /// </summary>
        /// <param name="tensors">The initial tensors.</param>
        public TensorReverse(Tensor[] tensors)
        {
            this.InitialTensors = tensors;
        }

        /// <summary>
        /// Gets the initial tensors.
        /// </summary>
        public Tensor[] InitialTensors { get; private set; }

        /// <summary>
        /// Computes the reverse gradient for the slice operation on an array of tensors.
        /// </summary>
        /// <param name="upstreamGradients">The gradients flowing from the upstream layers.</param>
        /// <param name="tensors">The original input tensors.</param>
        /// <param name="sliceSizes">The slice sizes used in the forward pass.</param>
        /// <returns>The gradients with respect to the input tensors.</returns>
        public static Tensor[] Slice3DTensorsReverse(Tensor[] upstreamGradients, Tensor[] tensors, int[] sliceSizes)
        {
            if (sliceSizes.Length != 3)
            {
                throw new ArgumentException("Slice sizes must be a 3-tuple.");
            }

            if (upstreamGradients.Length != tensors.Length)
            {
                throw new ArgumentException("The number of upstream gradients must match the number of tensors.");
            }

            var inputGradients = tensors.Select(tensor => new Tensor(tensor.Shape)).ToList();

            for (int t = 0; t < tensors.Length; t++)
            {
                var tensor = tensors[t];
                var upstreamGradient = upstreamGradients[t];

                if (tensor.Shape.Length != 3)
                {
                    throw new ArgumentException("All input tensors must be 3-dimensional.");
                }

                int[] numSlices = tensor.CalculateNumberOfSlices(sliceSizes);
                List<int[]> indicesList = tensor.GenerateSliceStartIndices(numSlices);

                int index = 0;
                foreach (var start in indicesList)
                {
                    for (int i = 0; i < start.Length; i++)
                    {
                        start[i] *= sliceSizes[i];
                    }

                    var sliceGradient = upstreamGradient.Slice(start, sliceSizes);
                    var reverseGradient = tensor.SliceReverse(sliceGradient, start, sliceSizes);

                    // Accumulate the gradients
                    for (int i = 0; i < reverseGradient.Data.Length; i++)
                    {
                        inputGradients[t].Data[i] += reverseGradient.Data[i];
                    }

                    index++;
                }
            }

            return inputGradients.ToArray();
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise addition.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseAddReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseAddReverse expects exactly two initial tensors.");
            }

            Tensor tensorA = this.InitialTensors[0];
            Tensor tensorB = this.InitialTensors[1];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorB, upstreamGradient);

            // The gradient is the same for both input tensors
            Tensor gradA = upstreamGradient;
            Tensor gradB = upstreamGradient;

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise absolute value function (abs).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor AbsReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];
            this.CheckShapeCompatibility(inputTensor, upstreamGradient);

            // Gradient of abs(x) is 1 for x >= 0, -1 for x < 0
            Tensor gradInput = new Tensor(inputTensor.Shape);
            Parallel.For(0, inputTensor.Data.Length, i =>
            {
                gradInput.Data[i] = inputTensor.Data[i] >= 0 ? upstreamGradient.Data[i] : -upstreamGradient.Data[i];
            });

            return gradInput;
        }

        /// <summary>
        /// Computes the gradient with respect to the input of a broadcast operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient from the next layer (after broadcasting).</param>
        /// <param name="originalShape">The original shape of the tensor before broadcasting.</param>
        /// <returns>The gradient with respect to the input tensor, matching the original shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the shapes are incompatible.</exception>
        public Tensor BroadcastToReverse(Tensor upstreamGradient, int[] originalShape)
        {
            // Ensure the upstream gradient shape is compatible with the original shape
            if (upstreamGradient.Shape.Length < originalShape.Length)
            {
                throw new ArgumentException("Upstream gradient shape must have the same rank or higher than the original shape.");
            }

            // Pad the original shape to match the rank of the upstream gradient
            int[] paddedOriginalShape = new int[upstreamGradient.Shape.Length];
            int padOffset = upstreamGradient.Shape.Length - originalShape.Length;

            for (int i = 0; i < upstreamGradient.Shape.Length; i++)
            {
                if (i < padOffset)
                {
                    paddedOriginalShape[i] = 1; // Pad with 1s
                }
                else
                {
                    paddedOriginalShape[i] = originalShape[i - padOffset];
                }
            }

            // Ensure shapes are compatible
            for (int i = 0; i < upstreamGradient.Shape.Length; i++)
            {
                if (paddedOriginalShape[i] != 1 && paddedOriginalShape[i] != upstreamGradient.Shape[i])
                {
                    // If the dimension of the upstream gradient is not divisible by the original shape (and it's not broadcasting), it's invalid
                    if (upstreamGradient.Shape[i] % paddedOriginalShape[i] != 0)
                    {
                        throw new ArgumentException($"Shapes are not compatible for broadcasting at dimension {i}.");
                    }
                }
            }

            // For the reverse operation, we need to sum gradients when a dimension was expanded
            var resultSize = originalShape.Aggregate(1, (a, b) => a * b);
            var result = PradTools.AllocateArray(resultSize);

            // For simple thread safety, use local partitioning instead of thread-local arrays
            object lockObj = new object();
            Parallel.ForEach(
                Partitioner.Create(0, upstreamGradient.Data.Length),
                range =>
                {
                    // Create a local accumulator for this range
                    var localResult = PradTools.AllocateArray(resultSize);

                    // Process this chunk of indices
                    for (int gradIndex = range.Item1; gradIndex < range.Item2; gradIndex++)
                    {
                        // Find the corresponding index in the original tensor
                        int originalIndex = this.GetReverseIndex(gradIndex, upstreamGradient.Shape, paddedOriginalShape, originalShape);

                        // Accumulate in local array without synchronization
                        localResult[originalIndex] += upstreamGradient.Data[gradIndex];
                    }

                    // Once the local chunk is processed, add results to the main array under a lock
                    lock (lockObj)
                    {
                        for (int i = 0; i < resultSize; i++)
                        {
                            result[i] += localResult[i];
                        }
                    }
                });

            return new Tensor(originalShape, result);
        }

        /// <summary>
        /// Maps an index from the broadcasted gradient back to the original tensor for gradient accumulation.
        /// </summary>
        /// <param name="gradIndex">Index in the gradient tensor.</param>
        /// <param name="gradShape">Shape of the gradient tensor.</param>
        /// <param name="paddedOriginalShape">Padded original shape.</param>
        /// <param name="originalShape">The original tensor shape.</param>
        /// <returns>The index in the original tensor where this gradient should be accumulated.</returns>
        public int GetReverseIndex(int gradIndex, int[] gradShape, int[] paddedOriginalShape, int[] originalShape)
        {
            // Convert gradient index to multi-dimensional coordinates
            int[] gradCoords = new int[gradShape.Length];
            int remainingIndex = gradIndex;
            for (int i = gradShape.Length - 1; i >= 0; i--)
            {
                gradCoords[i] = remainingIndex % gradShape[i];
                remainingIndex /= gradShape[i];
            }

            // Map to coordinates in the original tensor, accounting for broadcasting
            int[] originalCoords = new int[originalShape.Length];
            int padOffset = gradShape.Length - originalShape.Length;

            for (int i = 0; i < originalShape.Length; i++)
            {
                int gradDim = i + padOffset;

                if (paddedOriginalShape[gradDim] == 1)
                {
                    // This dimension was broadcast from 1 to n - always use index 0
                    originalCoords[i] = 0;
                }
                else if (gradShape[gradDim] > paddedOriginalShape[gradDim] &&
                         gradShape[gradDim] % paddedOriginalShape[gradDim] == 0)
                {
                    // This dimension was broadcast from n to m where m > n and m % n == 0
                    // We need to wrap to the correct original index
                    originalCoords[i] = gradCoords[gradDim] % paddedOriginalShape[gradDim];
                }
                else
                {
                    // Normal case - dimensions match exactly
                    originalCoords[i] = gradCoords[gradDim];
                }
            }

            // Convert back to flat index
            int originalIndex = 0;
            int stride = 1;
            for (int i = originalShape.Length - 1; i >= 0; i--)
            {
                originalIndex += originalCoords[i] * stride;
                stride *= originalShape[i];
            }

            return originalIndex;
        }

        /// <summary>
        /// Maps the original flattened index to the broadcasted tensor's flattened index.
        /// </summary>
        /// <param name="flatIndex">The original flattened index.</param>
        /// <param name="originalShape">The original shape of the tensor.</param>
        /// <param name="paddedOriginalShape">The padded original shape for broadcasting.</param>
        /// <param name="broadcastShape">The target broadcasted shape.</param>
        /// <returns>The corresponding index in the broadcasted data array.</returns>
        public int GetBroadcastIndex(int flatIndex, int[] originalShape, int[] paddedOriginalShape, int[] broadcastShape)
        {
            // Convert flatIndex to multi-dimensional coordinates for the original shape
            int[] originalCoords = new int[originalShape.Length];
            int remainingIndex = flatIndex;
            for (int i = originalShape.Length - 1; i >= 0; i--)
            {
                originalCoords[i] = remainingIndex % originalShape[i];
                remainingIndex /= originalShape[i];
            }

            // Map originalCoords to newCoords for the broadcasted tensor
            int[] broadcastCoords = new int[broadcastShape.Length];
            int padOffset = broadcastShape.Length - originalShape.Length;

            for (int i = 0; i < broadcastShape.Length; i++)
            {
                if (i < padOffset)
                {
                    broadcastCoords[i] = 0; // Broadcasting over this dimension
                }
                else
                {
                    if (paddedOriginalShape[i] == 1)
                    {
                        // Dimension is broadcast from 1 to n
                        broadcastCoords[i] = 0;
                    }
                    else if (broadcastShape[i] > paddedOriginalShape[i] &&
                             broadcastShape[i] % paddedOriginalShape[i] == 0)
                    {
                        // Handle case where broadcasting is by repeating (not just from size 1)
                        // For this forward mapping, we just map to the first occurrence
                        broadcastCoords[i] = originalCoords[i - padOffset];
                    }
                    else
                    {
                        // Normal case - dimensions match
                        broadcastCoords[i] = originalCoords[i - padOffset];
                    }
                }
            }

            // Convert newCoords (broadcasted shape) back to a flat index
            int broadcastFlatIndex = 0;
            int stride = 1;
            for (int i = broadcastShape.Length - 1; i >= 0; i--)
            {
                broadcastFlatIndex += broadcastCoords[i] * stride;
                stride *= broadcastShape[i];
            }

            return broadcastFlatIndex;
        }

        /// <summary>
        /// Computes the gradient of the modified Bessel function of the first kind, I0(x), with respect to the input,
        /// and scales it by the upstream gradient.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the next layer in the backward pass.</param>
        /// <returns>The gradient of the input tensor with respect to I0(x), scaled by the upstream gradient.</returns>
        public Tensor BesselI0Reverse(Tensor upstreamGradient)
        {
            var initial = this.InitialTensors[0];

            // Ensure the upstream gradient has the same shape as the input tensor
            if (!initial.Shape.SequenceEqual(upstreamGradient.Shape))
            {
                throw new ArgumentException("The shape of the upstream gradient must match the shape of the input tensor.");
            }

            // Create a tensor to store the gradient of the input with respect to the output
            var inputGradient = new Tensor(initial.Shape);

            // Precompute constants for the small and large x approximations
            const double threshold = 3.75;
            const double invThreshold = 1 / threshold;
            const double sqrt2pi = 0.39894228;

            // Coefficients for small x approximation
            double[] smallGradCoeffs = { 7.0312458, 9.2698272, 4.8269968, 1.3298928, 0.2162304, 0.0320391 };

            // Coefficients for large x approximation (precomputed derivatives of the Bessel function)
            double[] largeGradCoeffs = { 0.01328592, 0.00450638, -0.00472695, 0.03665124, -0.10288530, 0.15813222, -0.11533431, 0.03139016 };

            for (int i = 0; i < initial.Data.Length; i++)
            {
                double x = initial.Data[i];
                double absX = Math.Abs(x);

                // Compute the gradient of I0(x) with respect to x
                if (absX < threshold)
                {
                    // Small x approximation for the gradient
                    double t = x * invThreshold;
                    double t2 = t * t;
                    double dI0_dx = (x * invThreshold) *
                                    (smallGradCoeffs[0] + (t2 * (smallGradCoeffs[1] + (t2 * (smallGradCoeffs[2] +
                                    (t2 * (smallGradCoeffs[3] + (t2 * (smallGradCoeffs[4] + (t2 * smallGradCoeffs[5]))))))))));

                    // Multiply by the upstream gradient
                    inputGradient.Data[i] = PradTools.Cast(dI0_dx * upstreamGradient.Data[i]);
                }
                else
                {
                    // Large x approximation for the gradient
                    double t = threshold / absX;
                    double expTerm = Math.Exp(absX) / Math.Sqrt(absX);

                    // Use precomputed coefficients for the large x derivative
                    double besselTerm = sqrt2pi +
                                        (t * (largeGradCoeffs[0] + (t * (largeGradCoeffs[1] + (t * (largeGradCoeffs[2] +
                                        (t * (largeGradCoeffs[3] + (t * (largeGradCoeffs[4] + (t * (largeGradCoeffs[5] +
                                        (t * (largeGradCoeffs[6] + (t * largeGradCoeffs[7])))))))))))))));

                    double gradientTerm = expTerm * (besselTerm -
                                        (0.5 * t * (largeGradCoeffs[0] + (t * (largeGradCoeffs[1] + (t * (largeGradCoeffs[2] +
                                        (t * (largeGradCoeffs[3] + (t * (largeGradCoeffs[4] + (t * (largeGradCoeffs[5] +
                                        (t * (largeGradCoeffs[6] + (t * largeGradCoeffs[7]))))))))))))))));

                    // Adjust the sign based on the original x and scale by upstream gradient
                    inputGradient.Data[i] = PradTools.Cast(gradientTerm * Math.Sign(x) * upstreamGradient.Data[i]);
                }
            }

            return inputGradient;
        }

        /// <summary>
        /// Computes the gradient with respect to the input of a summation operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient from the next layer (after summation).</param>
        /// <param name="axes">The axes along which the original sum operation was performed.</param>
        /// <returns>The gradient with respect to the input tensor, matching the original shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the shapes are incompatible.</exception>
        public Tensor SumReverse(Tensor upstreamGradient, int[] axes)
        {
            var initial = this.InitialTensors[0];

            if (axes.Length == 1 && axes[0] == -1)
            {
                axes[0] = initial.Shape.Length - 1;
            }

            // Step 1: Create a new tensor with the same shape as the original input
            var gradientTensor = new Tensor(initial.Shape, PradTools.AllocateArray(initial.Data.Length));

            // Step 2: Broadcast the upstream gradient to the shape of the original input
            int[] broadcastDims = Enumerable.Range(0, initial.Shape.Length)
                                            .Where(i => !axes.Contains(i))
                                            .ToArray();

            // Step 3: Populate the gradient tensor
            int[] currentIndices = new int[initial.Shape.Length];
            this.PopulateGradientRecursive(upstreamGradient, gradientTensor, axes, broadcastDims, 0, currentIndices);

            return gradientTensor;
        }

        /// <summary>
        /// Populate.
        /// </summary>
        /// <param name="upstreamGradient">A.</param>
        /// <param name="gradientTensor">B.</param>
        /// <param name="sumAxes">C.</param>
        /// <param name="broadcastDims">D.</param>
        /// <param name="depth">E.</param>
        /// <param name="currentIndices">F.</param>
        public void PopulateGradientRecursive(Tensor upstreamGradient, Tensor gradientTensor, int[] sumAxes, int[] broadcastDims, int depth, int[] currentIndices)
        {
            var initial = this.InitialTensors[0];
            if (depth == initial.Shape.Length)
            {
                int gradientIndex = 0;
                int upstreamIndex = 0;
                for (int i = 0; i < initial.Shape.Length; i++)
                {
                    gradientIndex += currentIndices[i] * gradientTensor.Strides[i];
                    if (broadcastDims.Contains(i))
                    {
                        upstreamIndex += currentIndices[i] * upstreamGradient.Strides[broadcastDims.ToList().IndexOf(i)];
                    }
                }

                gradientTensor.Data[gradientIndex] = upstreamGradient.Data[upstreamIndex];
                return;
            }

            for (int i = 0; i < initial.Shape[depth]; i++)
            {
                currentIndices[depth] = i;
                this.PopulateGradientRecursive(upstreamGradient, gradientTensor, sumAxes, broadcastDims, depth + 1, currentIndices);
            }
        }

        /// <summary>
        /// Recursively broadcasts the upstream gradient back to the original shape.
        /// </summary>
        /// <param name="depth">Current recursion depth (axis).</param>
        /// <param name="currentIndices">Current indices into the original tensor.</param>
        /// <param name="originalStrides">Strides for the original tensor.</param>
        /// <param name="upstreamStrides">Strides for the upstream gradient.</param>
        /// <param name="upstreamData">The data array of the upstream gradient.</param>
        /// <param name="inputGradient">The input gradient being filled.</param>
        public void BroadcastUpstreamGradient(int depth, int[] currentIndices, int[] originalStrides, int[] upstreamStrides, double[] upstreamData, double[] inputGradient)
        {
            if (depth == originalStrides.Length)
            {
                // Calculate original tensor index
                int originalIndex = 0;
                for (int i = 0; i < currentIndices.Length; i++)
                {
                    originalIndex += currentIndices[i] * originalStrides[i];
                }

                // Calculate upstream tensor index
                int upstreamIndex = 0;
                for (int i = 0; i < upstreamStrides.Length; i++)
                {
                    if (upstreamStrides[i] != 0)
                    {
                        upstreamIndex += (currentIndices[i] % upstreamStrides[i]) * upstreamStrides[i];
                    }
                }

                // Broadcast the upstream gradient value to the input gradient
                inputGradient[originalIndex] += upstreamData[upstreamIndex];
                return;
            }

            // Recur into each element along this dimension
            for (int i = 0; i < originalStrides[depth]; i++)
            {
                currentIndices[depth] = i;
                this.BroadcastUpstreamGradient(depth + 1, currentIndices, originalStrides, upstreamStrides, upstreamData, inputGradient);
            }
        }

        /// <summary>
        /// Computes strides for a given shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>An array of strides.</returns>
        public int[] CalculateStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            strides[strides.Length - 1] = 1;
            for (int i = strides.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            return strides;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise logarithm (base 10).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor LogReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];
            this.CheckShapeCompatibility(inputTensor, upstreamGradient);

            // Gradient of log(x) is 1/x
            Tensor gradInput = new Tensor(inputTensor.Shape);
            Vml.Inv(inputTensor.Data.Length, inputTensor.Data, gradInput.Data);

            // Multiply by upstream gradient
            Vml.Mul(gradInput.Data.Length, gradInput.Data, upstreamGradient.Data, gradInput.Data);

            return gradInput;
        }

        /// <summary>
        /// Computes the gradient for the tanh operation.
        /// The derivative of tanh(x) is 1 - tanh^2(x).
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <returns>The gradient with respect to the input.</returns>
        public Tensor ElementwiseTanhReverse(Tensor upstreamGradient)
        {
            var input = this.InitialTensors[0];
            var tanhValue = input.ElementwiseTanh();
            var squaredTanh = tanhValue.ElementwiseMultiply(tanhValue);
            var derivative = new Tensor(input.Shape, PradTools.One).ElementwiseSub(squaredTanh);
            return derivative.ElementwiseMultiply(upstreamGradient);
        }

        /// <summary>
        /// Computes the gradient for the leaky ReLU operation.
        /// The derivative is 1 for x > 0, alpha for x ≤ 0.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <param name="alpha">The slope for negative values.</param>
        /// <returns>The gradient with respect to the input.</returns>
        public Tensor ElementwiseLeakyReLUReverse(Tensor upstreamGradient, double alpha)
        {
            var input = this.InitialTensors[0];
            var gradient = new Tensor(input.Shape);

            // Vector size for the current architecture
            int vectorSize = PradTools.VectorCount();

            // Prepare vectorized constants
            var alphaVector = PradTools.AllocateVector(PradTools.Cast(alpha));
            var zeroVector = PradTools.VectorZero();
            var oneVector = PradTools.VectorOne();

            // Determine chunk size for parallelization
            int chunkSize = Math.Max(vectorSize * 1000, 1); // Process at least 1000 vectors per thread

            Parallel.For(0, (input.Data.Length + chunkSize - 1) / chunkSize, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, input.Data.Length);
                int i = start;

                // Process vectors within this chunk
                for (; i <= end - vectorSize; i += vectorSize)
                {
                    var inputVec = PradTools.AllocateVector(input.Data, i);
                    var upstreamVec = PradTools.AllocateVector(upstreamGradient.Data, i);

                    var mask = Vector.GreaterThan(inputVec, zeroVector);
                    var maskDouble = Vector.ConditionalSelect(mask, oneVector, alphaVector);

                    var result = upstreamVec * maskDouble;
                    result.CopyTo(gradient.Data, i);
                }

                // Handle remaining elements in this chunk
                for (; i < end; i++)
                {
                    gradient.Data[i] = input.Data[i] > 0 ?
                        upstreamGradient.Data[i] :
                        PradTools.Cast(alpha) * upstreamGradient.Data[i];
                }
            });

            return gradient;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise natural logarithm (ln).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor LnReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];
            this.CheckShapeCompatibility(inputTensor, upstreamGradient);

            // Gradient of ln(x) is 1/x
            Tensor gradInput = new Tensor(inputTensor.Shape);
            Vml.Inv(inputTensor.Data.Length, inputTensor.Data, gradInput.Data);

            // Multiply by upstream gradient
            Vml.Mul(gradInput.Data.Length, gradInput.Data, upstreamGradient.Data, gradInput.Data);

            return gradInput;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise exponential function (exp).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ExpReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];
            this.CheckShapeCompatibility(inputTensor, upstreamGradient);

            // Gradient of exp(x) is exp(x)
            Tensor gradInput = new Tensor(inputTensor.Shape);
            Vml.Exp(inputTensor.Data.Length, inputTensor.Data, gradInput.Data);

            // Multiply by upstream gradient
            Vml.Mul(gradInput.Data.Length, gradInput.Data, upstreamGradient.Data, gradInput.Data);

            return gradInput;
        }

        /// <summary>
        /// Computes the reverse gradient for the SelfPair operation, optimized with precomputed offsets.
        /// </summary>
        /// <param name="upstreamGradient">The gradient tensor from the output of SelfPair, shape [2, M].</param>
        /// <returns>The gradient with respect to the original input tensor, shape [1, N].</returns>
        public Tensor SelfPairReverse(Tensor upstreamGradient)
        {
            Tensor originalTensor = this.InitialTensors[0];

            if (upstreamGradient.Shape.Length != 2 || upstreamGradient.Shape[0] != 2)
            {
                throw new ArgumentException("Upstream gradient must be of shape [2, M].");
            }

            if (originalTensor.Shape.Length != 2 || originalTensor.Shape[0] != 1)
            {
                throw new ArgumentException("Original tensor must be of shape [1, N].");
            }

            int n = originalTensor.Shape[1];
            int m = n * (n - 1) / 2; // Expected number of pairs

            if (upstreamGradient.Shape[1] != m)
            {
                throw new ArgumentException("Upstream gradient does not have the correct shape based on input size.");
            }

            // Create the output gradient tensor of shape [1, N]
            var resultGradient = new Tensor(new int[] { 1, n });
            var resultData = resultGradient.Data;

            // Precompute offsets for each index based on cumulative pair count
            int[] offsets = new int[n];
            for (int i = 1; i < n; i++)
            {
                offsets[i] = offsets[i - 1] + (n - i);
            }

            // Accumulate gradients in parallel
            Parallel.For(0, n - 1, i =>
            {
                int offset = offsets[i];
                for (int j = i + 1; j < n; j++)
                {
                    // For pair (i, j), add the upstream gradients to resultData[i] and resultData[j]
                    resultData[i] += upstreamGradient.Data[offset];
                    resultData[j] += upstreamGradient.Data[m + offset];
                    offset++;
                }
            });

            return resultGradient;
        }

        /// <summary>
        /// Computes the gradients for the embedding tensor and the sparsity tensor from the upstream gradient tensor.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient tensor with the same shape as the output of OnOffEmbedding.</param>
        /// <param name="indices">Tensor of indices, representing which rows were selected in the forward pass.</param>
        /// <param name="binaryCondition">Tensor with binary values (0 or 1) indicating the condition for each index in the forward pass.</param>
        /// <returns>A tuple containing the gradients for the embedding tensor and the sparsity tensor.</returns>
        /// <exception cref="ArgumentException">Thrown if upstreamGradient shape is incompatible or if binaryCondition contains values other than 0 or 1.</exception>
        public (Tensor embeddingGradient, Tensor sparsityGradient) OnOffEmbeddingReverse(Tensor upstreamGradient, Tensor indices, Tensor binaryCondition)
        {
            Tensor original = this.InitialTensors[0];

            // Validate input shapes
            if (!indices.Shape.SequenceEqual(binaryCondition.Shape) || indices.Shape.Length + 1 != upstreamGradient.Shape.Length)
            {
                throw new ArgumentException("Indices and binary condition tensors must match in shape, and upstream gradient shape must align with the output of OnOffEmbedding.");
            }

            if (upstreamGradient.Shape[^1] != original.Shape[1] * 2)
            {
                throw new ArgumentException("Upstream gradient tensor must have twice the embedding size as the last dimension.");
            }

            int embeddingSize = original.Shape[1];
            int newEmbeddingSize = embeddingSize * 2;

            // Initialize gradient tensors
            var embeddingGradient = new Tensor(original.Shape);
            var sparsityGradient = new Tensor(new int[] { 1, embeddingSize });

            // Accumulate gradients for each index
            Parallel.For(0, indices.Data.Length, i =>
            {
                int index = (int)indices.Data[i];
                int binary = (int)binaryCondition.Data[i];

                // Locate the corresponding gradient rows
                int embeddingRowStart = index * embeddingSize;
                int sparsityRowStart = 0;  // Single row for sparsity, so always start at 0

                // Upstream gradient row for doubled columns
                int gradientRowStart = i * newEmbeddingSize;

                for (int j = 0; j < embeddingSize; j++)
                {
                    if (binary == 0)
                    {
                        // For binary 0: upstream gradient for embedding in even indices, sparsity in odd indices
                        embeddingGradient.Data[embeddingRowStart + j] += upstreamGradient.Data[gradientRowStart + (2 * j)];
                        sparsityGradient.Data[sparsityRowStart + j] += upstreamGradient.Data[gradientRowStart + (2 * j) + 1];
                    }
                    else
                    {
                        // For binary 1: upstream gradient for sparsity in even indices, embedding in odd indices
                        sparsityGradient.Data[sparsityRowStart + j] += upstreamGradient.Data[gradientRowStart + (2 * j)];
                        embeddingGradient.Data[embeddingRowStart + j] += upstreamGradient.Data[gradientRowStart + (2 * j) + 1];
                    }
                }
            });

            return (embeddingGradient, sparsityGradient);
        }

        /// <summary>
        /// Computes the reverse gradient for the MultiplyColumns operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient tensor from the output of MultiplyColumns, shape [1, P].</param>
        /// <returns>The gradient with respect to the original input tensor, shape [M, P].</returns>
        public Tensor MultiplyColumnsReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];

            if (inputTensor.Shape.Length != 2 || upstreamGradient.Shape.Length != 2)
            {
                throw new ArgumentException("MultiplyColumnsReverse requires a 2D input tensor and a 2D upstream gradient tensor.");
            }

            int rows = inputTensor.Shape[0];
            int columns = inputTensor.Shape[1];

            if (upstreamGradient.Shape[0] != 1 || upstreamGradient.Shape[1] != columns)
            {
                throw new ArgumentException("Upstream gradient must have shape [1, columns] to match the number of columns in the input tensor.");
            }

            // Create the output gradient tensor of the same shape as the original input tensor
            var inputGradient = new Tensor(inputTensor.Shape);

            // Compute the column products from the original tensor
            var columnProducts = PradTools.AllocateArray(columns);
            for (int col = 0; col < columns; col++)
            {
                columnProducts[col] = 1;
                for (int row = 0; row < rows; row++)
                {
                    columnProducts[col] *= inputTensor.Data[(row * columns) + col];
                }
            }

            // Calculate the gradient for each element in the original tensor
            Parallel.For(0, columns, col =>
            {
                for (int row = 0; row < rows; row++)
                {
                    // Gradient for element (row, col) is upstreamGradient[col] * (column product / element)
                    var elementValue = inputTensor.Data[(row * columns) + col];
                    inputGradient.Data[(row * columns) + col] = upstreamGradient.Data[col] * (columnProducts[col] / elementValue);
                }
            });

            return inputGradient;
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise multiplication.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseMultiplyReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseMultiplyReverse expects exactly two initial tensors.");
            }

            Tensor tensorA = this.InitialTensors[0];
            Tensor tensorB = this.InitialTensors[1];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorB, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Tensor gradB = new Tensor(tensorB.Shape);

            Vml.Mul(tensorA.Data.Length, upstreamGradient.Data, tensorB.Data, gradA.Data);
            Vml.Mul(tensorB.Data.Length, upstreamGradient.Data, tensorA.Data, gradB.Data);

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise multiplication.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="mapping">The broadcast mapping.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseMultiplyBroadcastingReverse(Tensor upstreamGradient, BroadcastMapping mapping)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseMultiplyReverse expects exactly two initial tensors.");
            }

            var originalA = this.InitialTensors[0];
            var originalB = this.InitialTensors[1];

            var gradA = new Tensor(originalA.Shape);
            var gradB = new Tensor(originalB.Shape);

            int vectorSize = PradTools.VectorCount();
            int totalElements = upstreamGradient.Data.Length;

            // Fast path for identical shapes
            if (originalA.Shape.SequenceEqual(originalB.Shape) &&
                originalA.Shape.SequenceEqual(upstreamGradient.Shape))
            {
                for (int i = 0; i <= totalElements - vectorSize; i += vectorSize)
                {
                    var upstreamVec = PradTools.AllocateVector(upstreamGradient.Data, i);
                    var aVec = PradTools.AllocateVector(originalA.Data, i);
                    var bVec = PradTools.AllocateVector(originalB.Data, i);

                    (upstreamVec * bVec).CopyTo(gradA.Data, i);
                    (upstreamVec * aVec).CopyTo(gradB.Data, i);
                }

                // Handle remaining elements
                for (int i = totalElements - (totalElements % vectorSize); i < totalElements; i++)
                {
                    gradA.Data[i] = upstreamGradient.Data[i] * originalB.Data[i];
                    gradB.Data[i] = upstreamGradient.Data[i] * originalA.Data[i];
                }

                return new Tensor[] { gradA, gradB };
            }

            // Process each output position separately to avoid race conditions
            Parallel.For(0, originalA.Data.Length, outputIdx =>
            {
                // Find indices that contribute to this output position
                var indicesA = new List<int>();
                var indicesB = new List<int>();

                // Gather contributing indices
                for (int k = 0; k < mapping.SourceIndicesA.Length; k++)
                {
                    if (mapping.SourceIndicesA[k] == outputIdx)
                    {
                        indicesA.Add(k);
                    }

                    if (mapping.SourceIndicesB[k] == outputIdx)
                    {
                        indicesB.Add(k);
                    }
                }

                // Process gradA contributions using SIMD
                if (indicesA.Count > 0)
                {
                    var sumA = PradTools.VectorZero();
                    int vectorCount = indicesA.Count / vectorSize;

                    // Process vectors
                    for (int i = 0; i < vectorCount * vectorSize; i += vectorSize)
                    {
                        var upstreamValues = PradTools.AllocateArray(vectorSize);
                        var bValues = PradTools.AllocateArray(vectorSize);

                        for (int j = 0; j < vectorSize; j++)
                        {
                            int idx = indicesA[i + j];
                            upstreamValues[j] = upstreamGradient.Data[idx];
                            bValues[j] = originalB.Data[mapping.SourceIndicesB[idx]];
                        }

                        var upstreamVec = PradTools.AllocateVector(upstreamValues);
                        var bVec = PradTools.AllocateVector(bValues);
                        sumA += upstreamVec * bVec;
                    }

                    // Sum vector elements
                    var scalarSumA = PradTools.Zero;
                    for (int i = 0; i < vectorSize; i++)
                    {
                        scalarSumA += sumA[i];
                    }

                    // Handle remaining elements
                    for (int i = vectorCount * vectorSize; i < indicesA.Count; i++)
                    {
                        int idx = indicesA[i];
                        scalarSumA += upstreamGradient.Data[idx] * originalB.Data[mapping.SourceIndicesB[idx]];
                    }

                    gradA.Data[outputIdx] = scalarSumA;
                }

                // Process gradB contributions using SIMD
                if (indicesB.Count > 0)
                {
                    var sumB = PradTools.VectorZero();
                    int vectorCount = indicesB.Count / vectorSize;

                    // Process vectors
                    for (int i = 0; i < vectorCount * vectorSize; i += vectorSize)
                    {
                        var upstreamValues = PradTools.AllocateArray(vectorSize);
                        var aValues = PradTools.AllocateArray(vectorSize);

                        for (int j = 0; j < vectorSize; j++)
                        {
                            int idx = indicesB[i + j];
                            upstreamValues[j] = upstreamGradient.Data[idx];
                            aValues[j] = originalA.Data[mapping.SourceIndicesA[idx]];
                        }

                        var upstreamVec = PradTools.AllocateVector(upstreamValues);
                        var aVec = PradTools.AllocateVector(aValues);
                        sumB += upstreamVec * aVec;
                    }

                    // Sum vector elements
                    var scalarSumB = PradTools.Zero;
                    for (int i = 0; i < vectorSize; i++)
                    {
                        scalarSumB += sumB[i];
                    }

                    // Handle remaining elements
                    for (int i = vectorCount * vectorSize; i < indicesB.Count; i++)
                    {
                        int idx = indicesB[i];
                        scalarSumB += upstreamGradient.Data[idx] * originalA.Data[mapping.SourceIndicesA[idx]];
                    }

                    gradB.Data[outputIdx] = scalarSumB;
                }
            });

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for concatenation along any axis, taking into account a custom ordering of tensors.
        /// If the ordering is null, the tensors will be processed in their original order.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axis">The axis along which the concatenation was performed.</param>
        /// <param name="ordering">An optional array representing the custom ordering of the tensors. If null, no reordering is done.</param>
        /// <returns>An array of tensors representing the gradients for each input tensor.</returns>
        /// <exception cref="ArgumentException">Thrown if tensor shapes are incompatible or ordering is invalid.</exception>
        public Tensor[] ConcatReverse(Tensor upstreamGradient, int axis, int[]? ordering = null)
        {
            int numTensors = this.InitialTensors.Length;

            Tensor[] tensorsToProcess;

            if (ordering != null)
            {
                if (ordering.Length != numTensors)
                {
                    throw new ArgumentException("The ordering array must be the same length as the number of tensors.");
                }

                // Reorder tensors based on the provided ordering
                tensorsToProcess = new Tensor[numTensors];
                for (int i = 0; i < ordering.Length; i++)
                {
                    if (ordering[i] < 0 || ordering[i] >= numTensors)
                    {
                        throw new ArgumentException("Invalid ordering index.");
                    }

                    tensorsToProcess[i] = this.InitialTensors[ordering[i]];
                }
            }
            else
            {
                // If no ordering is provided, use the tensors in their original order
                tensorsToProcess = this.InitialTensors;
            }

            Tensor[] gradients = new Tensor[numTensors];

            // Normalize axis and validate
            axis = this.NormalizeAxis(axis, upstreamGradient.Shape.Length);
            this.ValidateTensorsForConcatenation(axis);

            int offset = 0;

            // Iterate over the tensors (in the original or reordered order)
            for (int i = 0; i < numTensors; i++)
            {
                int[] tensorShape = tensorsToProcess[i].Shape;
                int sliceSize = tensorShape[axis];  // The size of the slice along the concatenation axis
                int totalElementsToCopy = tensorShape.Aggregate(1, (a, b) => a * b);  // Total elements for the tensor

                // Create storage for the gradient data for this tensor
                var gradData = PradTools.AllocateArray(totalElementsToCopy);

                // Calculate the number of elements per slice (all dimensions except the concatenation axis)
                int elementsPerSlice = this.GetElementsPerSlice(upstreamGradient.Shape, axis);

                // Copy the relevant slices for this tensor
                for (int slice = 0; slice < elementsPerSlice; slice++)
                {
                    // Calculate the start and end indices in the upstream gradient
                    int upstreamStartIndex = (slice * upstreamGradient.Shape[axis]) + offset;
                    int upstreamEndIndex = upstreamStartIndex + sliceSize;

                    // Copy the slice into the gradient data for this tensor
                    Array.Copy(upstreamGradient.Data, upstreamStartIndex, gradData, slice * sliceSize, sliceSize);
                }

                // Create the tensor for the gradient and add it to the array
                gradients[i] = new Tensor(tensorShape, gradData);

                // Update the offset to move to the next tensor's slice
                offset += sliceSize;
            }

            if (ordering != null)
            {
                // Second reordering: Map the gradients back to their original order
                Tensor[] finalGradients = new Tensor[numTensors];
                for (int i = 0; i < ordering.Length; i++)
                {
                    finalGradients[ordering[i]] = gradients[i];
                }

                return finalGradients;
            }

            return gradients;
        }

        /// <summary>
        /// Calculates the number of elements per slice along all dimensions except the concatenation axis.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="axis">The axis along which the concatenation occurred.</param>
        /// <returns>The number of elements per slice along the specified axis.</returns>
        public int GetElementsPerSlice(int[] shape, int axis)
        {
            // Multiply the dimensions except for the concatenation axis to get the number of slices
            return shape.Where((_, idx) => idx != axis).Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Extracts a slice of the gradient from the upstream gradient tensor along a specific axis.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient tensor.</param>
        /// <param name="offset">The starting offset in the upstream gradient.</param>
        /// <param name="sliceLength">The length of the slice along the specified axis.</param>
        /// <param name="gradShape">The shape of the resulting gradient tensor.</param>
        /// <param name="axis">The axis along which to slice.</param>
        /// <returns>A new tensor with the sliced gradient data.</returns>
        public Tensor ExtractGradientSlice(Tensor upstreamGradient, int offset, int sliceLength, int[] gradShape, int axis)
        {
            int[] upstreamShape = upstreamGradient.Shape;
            int rank = upstreamShape.Length;

            // Create the gradient data array to hold the extracted slice
            var gradData = PradTools.AllocateArray(gradShape.Aggregate(1, (a, b) => a * b));
            int stride = this.GetStride(upstreamShape, axis); // Calculate stride along the axis

            // Multi-dimensional extraction across the tensor
            int subTensorSize = gradShape.Skip(axis + 1).Aggregate(1, (a, b) => a * b); // Size of a single row or sub-dimension
            int gradIndex = 0;

            // Iterate over the specified axis, extracting each slice
            for (int i = 0; i < sliceLength; i++)
            {
                for (int j = 0; j < subTensorSize; j++)
                {
                    int upstreamIndex = (offset * stride) + (i * stride) + j;
                    gradData[gradIndex++] = upstreamGradient.Data[upstreamIndex];
                }
            }

            return new Tensor(gradShape, gradData);
        }

        /// <summary>
        /// Calculates the stride size for slicing along a specific axis.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="axis">The axis along which the stride is calculated.</param>
        /// <returns>The stride size for the axis.</returns>
        public int GetStride(int[] shape, int axis)
        {
            // Calculate the stride for the given axis by multiplying the sizes of all subsequent dimensions
            return shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Normalizes the axis to handle negative values.
        /// </summary>
        /// <param name="axis">The axis provided.</param>
        /// <param name="rank">The rank of the tensors.</param>
        /// <returns>Normalized axis.</returns>
        /// <exception cref="ArgumentException">Thrown if the axis is out of bounds.</exception>
        public int NormalizeAxis(int axis, int rank)
        {
            if (axis < 0)
            {
                axis += rank;
            }

            if (axis < 0 || axis >= rank)
            {
                throw new ArgumentException($"Axis value {axis} is out of bounds for tensor rank {rank}.");
            }

            return axis;
        }

        /// <summary>
        /// Validates that the tensors are compatible for concatenation along the given axis.
        /// </summary>
        /// <param name="axis">The axis along which concatenation was performed.</param>
        /// <exception cref="ArgumentException">Thrown if the tensors are not compatible for concatenation.</exception>
        public void ValidateTensorsForConcatenation(int axis)
        {
            int[] referenceShape = this.InitialTensors[0].Shape;
            for (int i = 1; i < this.InitialTensors.Length; i++)
            {
                int[] currentShape = this.InitialTensors[i].Shape;

                for (int dim = 0; dim < referenceShape.Length; dim++)
                {
                    if (dim != axis && referenceShape[dim] != currentShape[dim])
                    {
                        throw new ArgumentException("All tensor shapes must match except for the concatenation axis.");
                    }
                }
            }
        }

        /// <summary>
        /// Calculates the size of the "sub-tensor" for slicing, ignoring the given axis.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="axis">The axis along which the slicing is performed.</param>
        /// <returns>The size of the sub-tensor (product of dimensions except the axis).</returns>
        public int GetSubTensorSize(int[] shape, int axis)
        {
            return shape.Where((_, idx) => idx != axis).Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise square.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseSquareReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSquareReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Vml.Mul(tensorA.Data.Length, upstreamGradient.Data, tensorA.Data, gradA.Data);
            Blas.scal(PradTools.Two, gradA.Data);

            return gradA;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise power operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="exponent">The exponent used in the forward power operation. Can be a scalar or a tensor.</param>
        /// <returns>The gradient with respect to the input tensor and the exponent (if it's a tensor).</returns>
        public Tensor[] PowReverse(Tensor upstreamGradient, object exponent)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("PowReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];
            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            var gradX = new Tensor(tensorA.Shape);
            Tensor? gradExponent = null;

            if (exponent is double scalarExponent)
            {
                // Case 1: Scalar exponent
                // d/dx(x^n) = n * x^(n-1)
                var temp = new Tensor(tensorA.Shape);
                var exponentArray = PradTools.FillArray(tensorA.Data.Length, scalarExponent);
                var ones = PradTools.OneArray(tensorA.Data.Length);
                var resultTemp = PradTools.AllocateArray(tensorA.Data.Length);
                Vml.Sub(tensorA.Data.Length, exponentArray, ones, resultTemp);
                Vml.Pow(tensorA.Data.Length, tensorA.Data, resultTemp, temp.Data);
                Vml.Mul(gradX.Data.Length, exponentArray, temp.Data, gradX.Data);
                Vml.Mul(gradX.Data.Length, upstreamGradient.Data, gradX.Data, gradX.Data);
            }
            else if (exponent is float scalarExponentF)
            {
                // Case 1: Scalar exponent
                // d/dx(x^n) = n * x^(n-1)
                var temp = new Tensor(tensorA.Shape);
                var exponentArray = PradTools.AllocateArray(tensorA.Data.Length);
                Array.Fill(exponentArray, scalarExponentF);
                var ones = PradTools.OneArray(tensorA.Data.Length);
                var resultTemp = PradTools.AllocateArray(tensorA.Data.Length);
                Vml.Sub(tensorA.Data.Length, exponentArray, ones, resultTemp);
                Vml.Pow(tensorA.Data.Length, tensorA.Data, resultTemp, temp.Data);
                Vml.Mul(gradX.Data.Length, exponentArray, temp.Data, gradX.Data);
                Vml.Mul(gradX.Data.Length, upstreamGradient.Data, gradX.Data, gradX.Data);
            }
            else if (exponent is Tensor exponentTensor)
            {
                // Case 2: Tensor exponent
                this.CheckShapeCompatibility(tensorA, exponentTensor);
                gradExponent = new Tensor(tensorA.Shape);

                // d/dx(x^y) = y * x^(y-1)
                var temp1 = new Tensor(tensorA.Shape);
                var temp2 = new Tensor(tensorA.Shape);
                Vml.Sub(exponentTensor.Data.Length, exponentTensor.Data, PradTools.OneArray(exponentTensor.Data.Length), temp1.Data);
                Vml.Pow(tensorA.Data.Length, tensorA.Data, temp1.Data, temp2.Data);
                Vml.Mul(gradX.Data.Length, exponentTensor.Data, temp2.Data, gradX.Data);
                Vml.Mul(gradX.Data.Length, upstreamGradient.Data, gradX.Data, gradX.Data);

                // d/dy(x^y) = x^y * ln(x)
                Vml.Ln(tensorA.Data.Length, tensorA.Data, temp1.Data);
                Vml.Pow(tensorA.Data.Length, tensorA.Data, exponentTensor.Data, temp2.Data);
                Vml.Mul(gradExponent.Data.Length, temp1.Data, temp2.Data, gradExponent.Data);
                Vml.Mul(gradExponent.Data.Length, upstreamGradient.Data, gradExponent.Data, gradExponent.Data);
            }
            else
            {
                throw new ArgumentException("Exponent must be either a float, double, or a Tensor.");
            }

            return gradExponent != null ? new[] { gradX, gradExponent } : new[] { gradX };
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise max operation between two tensors.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="other">The other tensor used in the forward max operation.</param>
        /// <returns>The gradient with respect to both input tensors.</returns>
        public Tensor[] MaxReverse(Tensor upstreamGradient, Tensor other)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("MaxReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];

            // Ensure both tensors have the same shape
            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorA, other);

            // Allocate gradient tensors for both inputs
            var gradX = new Tensor(tensorA.Shape);
            var gradOther = new Tensor(tensorA.Shape);

            // Compute element-wise maximum
            var maxMask = new Tensor(tensorA.Shape);

            // maxMask will be 1 where this > other, 0 otherwise
            Parallel.For(0, tensorA.Data.Length, i =>
            {
                maxMask.Data[i] = tensorA.Data[i] > other.Data[i] ? 1 : 0;
            });

            // Compute gradients based on maxMask
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, maxMask.Data, gradX.Data);  // gradient for `tensorA`

            // Invert maxMask to get the gradient for `other`
            Vml.Sub(maxMask.Data.Length, PradTools.OneArray(maxMask.Data.Length), maxMask.Data, gradOther.Data);
            Vml.Mul(gradOther.Data.Length, upstreamGradient.Data, gradOther.Data, gradOther.Data); // gradient for `other`

            return new[] { gradX, gradOther };
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise max operation with a scalar.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="scalar">The scalar used in the forward max operation.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor MaxReverse(Tensor upstreamGradient, double scalar)
        {
            Tensor tensorA = this.InitialTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            // Allocate gradient tensor
            var gradX = new Tensor(tensorA.Shape);

            // Compute max mask where this tensor > scalar
            var maxMask = new Tensor(tensorA.Shape);
            Parallel.For(0, tensorA.Data.Length, i =>
            {
                maxMask.Data[i] = tensorA.Data[i] > scalar ? 1 : 0;
            });

            // Compute gradient
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, maxMask.Data, gradX.Data);

            return gradX;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise min operation between two tensors.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="other">The other tensor used in the forward min operation.</param>
        /// <returns>The gradient with respect to both input tensors.</returns>
        public Tensor[] MinReverse(Tensor upstreamGradient, Tensor other)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("MinReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];

            // Ensure both tensors have the same shape
            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorA, other);

            // Allocate gradient tensors for both inputs
            var gradX = new Tensor(tensorA.Shape);
            var gradOther = new Tensor(tensorA.Shape);

            // Compute element-wise minimum
            var minMask = new Tensor(tensorA.Shape);

            // minMask will be 1 where this < other, 0 otherwise
            Parallel.For(0, tensorA.Data.Length, i =>
            {
                minMask.Data[i] = tensorA.Data[i] < other.Data[i] ? 1 : 0;
            });

            // Compute gradients based on minMask
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, minMask.Data, gradX.Data);  // gradient for `tensorA`

            // Invert minMask to get the gradient for `other`
            Vml.Sub(minMask.Data.Length, PradTools.OneArray(minMask.Data.Length), minMask.Data, gradOther.Data);
            Vml.Mul(gradOther.Data.Length, upstreamGradient.Data, gradOther.Data, gradOther.Data); // gradient for `other`

            return new[] { gradX, gradOther };
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise min operation with a scalar.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="scalar">The scalar used in the forward min operation.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor MinReverse(Tensor upstreamGradient, double scalar)
        {
            Tensor tensorA = this.InitialTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            // Allocate gradient tensor
            var gradX = new Tensor(tensorA.Shape);

            // Compute min mask where this tensor < scalar
            var minMask = new Tensor(tensorA.Shape);
            Parallel.For(0, tensorA.Data.Length, i =>
            {
                minMask.Data[i] = tensorA.Data[i] < scalar ? 1 : 0;
            });

            // Compute gradient
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, minMask.Data, gradX.Data);

            return gradX;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise arccosine (acos).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseArcCosReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseArcCosReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];
            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Tensor denom = new Tensor(tensorA.Shape);
            Tensor oneTensor = new Tensor(tensorA.Shape, PradTools.One);

            // Compute 1 - x^2 in denom
            Vml.Pow(tensorA.Data.Length, tensorA.Data, PradTools.FillArray(tensorA.Data.Length, PradTools.Two), denom.Data);
            Vml.Sub(denom.Data.Length, oneTensor.Data, denom.Data, denom.Data);

            // Ensure numerical stability by adding a small epsilon to avoid division by zero
            Vml.MaxMag(denom.Data.Length, denom.Data, PradTools.FillArray(denom.Data.Length, PradTools.Epsilon10), denom.Data);

            // Compute sqrt(1 - x^2)
            Vml.Sqrt(denom.Data.Length, denom.Data, denom.Data);

            // Calculate gradient: gradA = -upstreamGradient / denom
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, denom.Data, gradA.Data);
            Blas.scal(PradTools.NegativeOne, gradA.Data);

            return gradA;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise square root operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseSquareRootReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSquareRootReverse expects exactly one initial tensor.");
            }

            Tensor x = this.InitialTensors[0]; // The input to the square root operation
            this.CheckShapeCompatibility(x, upstreamGradient);

            var gradX = new Tensor(x.Shape);
            var sqrtX = new Tensor(x.Shape);

            var epsilon = PradTools.Epsilon10;
            var normalizedSqrtX = new Tensor(x.Shape);
            var epsilonTensor = new Tensor(x.Shape, epsilon);

            // Compute sqrt(x)
            Vml.Sqrt(x.Data.Length, x.Data, sqrtX.Data);

            Vml.MaxMag(sqrtX.Data.Length, sqrtX.Data, epsilonTensor.Data, normalizedSqrtX.Data);

            // Compute gradX = upstreamGradient / (2 * sqrt(x))
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, normalizedSqrtX.Data, gradX.Data);
            Blas.scal(PradTools.Half, gradX.Data);

            return gradX;
        }

        /// <summary>
        /// Computes the reverse gradient for ExtractPatches using the original input tensor.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer (patches).</param>
        /// <param name="filterSize">The size of the sliding window [filter_height, filter_width].</param>
        /// <param name="strides">The strides for the sliding window [stride_height, stride_width].</param>
        /// <param name="padding">Padding type ('VALID' or 'SAME').</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ExtractPatchesReverse(Tensor upstreamGradient, int[] filterSize, int[] strides, string padding)
        {
            // Use the initial input tensor before patches were extracted
            Tensor inputTensor = this.InitialTensors[0];

            if (filterSize.Length != 2 || strides.Length != 2)
            {
                throw new ArgumentException("Filter size and strides must have 2 dimensions (height, width).");
            }

            if (padding == "SAME")
            {
                return this.ExtractPatchesSameReverse(upstreamGradient, filterSize, strides);
            }

            int batchSize = inputTensor.Shape[0];
            int inputHeight = inputTensor.Shape[1];
            int inputWidth = inputTensor.Shape[2];
            int channels = inputTensor.Shape.Length == 3 ? 1 : inputTensor.Shape[3];

            int filterHeight = filterSize[0];
            int filterWidth = filterSize[1];
            int strideHeight = strides[0];
            int strideWidth = strides[1];

            // Compute padding for SAME or VALID mode
            int padTop, padBottom, padLeft, padRight;

            if (padding == "VALID")
            {
                padTop = padBottom = padLeft = padRight = 0;
            }
            else
            {
                throw new ArgumentException("Unsupported padding type. Use 'VALID' or 'SAME'.");
            }

            // Initialize the gradient for the input (zeroed out)
            int paddedHeight = inputHeight + padTop + padBottom;
            int paddedWidth = inputWidth + padLeft + padRight;
            Tensor inputGradient = new Tensor(new int[] { batchSize, paddedHeight, paddedWidth, channels });

            // Unpack the upstream gradients (patches) and sum into the respective locations
            int outHeight = upstreamGradient.Shape[1];
            int outWidth = upstreamGradient.Shape[2];
            int patchSize = filterHeight * filterWidth;

            Parallel.For(0, batchSize, b =>
            {
                for (int i = 0; i < outHeight; i++)
                {
                    for (int j = 0; j < outWidth; j++)
                    {
                        int patchIndex = ((b * outHeight * outWidth) + (i * outWidth) + j) * patchSize * channels;

                        for (int h = 0; h < filterHeight; h++)
                        {
                            int srcY = (i * strideHeight) + h;
                            int srcXStart = j * strideWidth;

                            for (int w = 0; w < filterWidth; w++)
                            {
                                for (int c = 0; c < channels; c++)
                                {
                                    // Use the upstream gradient to distribute gradients
                                    int inputGradientOffset = (((b * paddedHeight * paddedWidth) + ((srcY * paddedWidth) + srcXStart + w)) * channels) + c;
                                    int upstreamOffset = patchIndex + (((h * filterWidth) + w) * channels) + c;

                                    inputGradient.Data[inputGradientOffset] += upstreamGradient.Data[upstreamOffset];
                                }
                            }
                        }
                    }
                }
            });

            // Remove padding from the inputGradient, not the initial tensor
            if (padTop != 0 || padBottom != 0 || padLeft != 0 || padRight != 0)
            {
                return this.RemovePadding(inputGradient, padTop, padBottom, padLeft, padRight);
            }

            return inputGradient;
        }

        /// <summary>
        /// Computes the reverse gradient for ExtractPatches using the original input tensor.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer (patches).</param>
        /// <param name="filterSize">The size of the sliding window [filter_height, filter_width].</param>
        /// <param name="strides">The strides for the sliding window [stride_height, stride_width].</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ExtractPatchesSameReverse(Tensor upstreamGradient, int[] filterSize, int[] strides)
        {
            Tensor inputTensor = this.InitialTensors[0].DeepClone();

            var isReallyExpanded = false;
            if (inputTensor.Shape.Length == 2)
            {
                inputTensor = inputTensor.ExpandDims(-1);
                isReallyExpanded = true;
            }

            var isExpanded = false;
            if (inputTensor.Shape.Length == 3)
            {
                inputTensor = inputTensor.ExpandDims(0);
                isExpanded = true;
            }

            int batchSize = inputTensor.Shape[0];
            int inputHeight = inputTensor.Shape[1];
            int inputWidth = inputTensor.Shape[2];
            int channels = inputTensor.Shape[3];

            int filterHeight = filterSize[0];
            int filterWidth = filterSize[1];
            int strideHeight = strides[0];
            int strideWidth = strides[1];

            int outHeight = upstreamGradient.Shape[1];
            int outWidth = upstreamGradient.Shape[2];

            Tensor inputGradient = new Tensor(new int[] { batchSize, inputHeight, inputWidth, channels });

            int padTop = (filterHeight - 1) / 2;
            int padBottom = filterHeight - 1 - padTop;
            int padLeft = (filterWidth - 1) / 2;
            int padRight = filterWidth - 1 - padLeft;

            Parallel.For(0, batchSize, b =>
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        for (int fh = 0; fh < filterHeight; fh++)
                        {
                            for (int fw = 0; fw < filterWidth; fw++)
                            {
                                int ih = (oh * strideHeight) + fh - padTop;
                                int iw = (ow * strideWidth) + fw - padLeft;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    for (int c = 0; c < channels; c++)
                                    {
                                        int upstreamIndex = (((((((((b * outHeight) + oh) * outWidth) + ow) * filterHeight) + fh) * filterWidth) + fw) * channels) + c;
                                        int inputGradientIndex = (((((b * inputHeight) + ih) * inputWidth) + iw) * channels) + c;

                                        inputGradient.Data[inputGradientIndex] += upstreamGradient.Data[upstreamIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            });

            if (isExpanded)
            {
                if (isReallyExpanded)
                {
                    return inputGradient.Reshape(inputGradient.Shape.Skip(1).Take(2).ToArray());
                }

                return inputGradient.Reshape(inputGradient.Shape.Skip(1).ToArray());
            }

            return inputGradient;
        }

        /// <summary>
        /// Removes padding from a padded tensor (used after reverse ExtractPatches).
        /// </summary>
        /// <param name="paddedTensor">The tensor with padding.</param>
        /// <param name="padTop">Padding on the top.</param>
        /// <param name="padBottom">Padding on the bottom.</param>
        /// <param name="padLeft">Padding on the left.</param>
        /// <param name="padRight">Padding on the right.</param>
        /// <returns>A tensor with padding removed.</returns>
        public Tensor RemovePadding(Tensor paddedTensor, int padTop, int padBottom, int padLeft, int padRight)
        {
            // Calculate the new shape after removing the padding
            int[] newShape =
            {
                paddedTensor.Shape[0],                                 // Batch size remains the same
                paddedTensor.Shape[1] - padTop - padBottom,            // Adjust the height by removing padding
                paddedTensor.Shape[2] - padLeft - padRight,            // Adjust the width by removing padding
                paddedTensor.Shape[3],                                 // Number of channels remains the same
            };

            // Create the new unpadded tensor
            Tensor unpadded = new Tensor(newShape);

            // Calculate the size of each row (in terms of number of elements) for both the padded and unpadded tensor
            int paddedRowSize = paddedTensor.Shape[2] * paddedTensor.Shape[3];     // Width * Channels for the padded tensor
            int unpaddedRowSize = newShape[2] * newShape[3];                       // Width * Channels for the unpadded tensor
            int channelSize = paddedTensor.Shape[3];                               // Number of channels is the same

            Parallel.For(0, paddedTensor.Shape[0], b =>
            {
                for (int i = padTop; i < paddedTensor.Shape[1] - padBottom; i++)
                {
                    // Calculate the source and destination indices for Buffer.BlockCopy
                    int srcOffset = ((((b * paddedTensor.Shape[1]) + i) * paddedTensor.Shape[2]) + padLeft) * channelSize;   // Skip the left padding in the row
                    int destOffset = (((b * newShape[1]) + (i - padTop)) * newShape[2]) * channelSize;                     // Place the row in the unpadded tensor

                    // Copy the entire row from padded to unpadded tensor
                    Buffer.BlockCopy(
                        paddedTensor.Data,
                        srcOffset * PradTools.SizeOf,              // Source data pointer
                        unpadded.Data,
                        destOffset * PradTools.SizeOf,                 // Destination data pointer
                        unpaddedRowSize * PradTools.SizeOf);
                }
            });

            return unpadded;
        }

        /// <summary>
        /// Computes the reverse gradient for the sigmoid activation.
        /// The derivative is sigmoid(x) * (1 - sigmoid(x)).
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor SigmoidReverse(Tensor upstreamGradient)
        {
            Tensor inputTensor = this.InitialTensors[0];
            this.CheckShapeCompatibility(inputTensor, upstreamGradient);

            // First compute sigmoid(x)
            Tensor sigmoid = inputTensor.Sigmoid();

            // Now compute 1 - sigmoid(x)
            Tensor oneMinusSigmoid = new Tensor(sigmoid.Shape);
            var ones = PradTools.AllocateArray(sigmoid.Data.Length);
            Array.Fill(ones, PradTools.One);
            Vml.Sub(sigmoid.Data.Length, ones, sigmoid.Data, oneMinusSigmoid.Data);

            // Then sigmoid(x) * (1 - sigmoid(x))
            Tensor sigmoidDerivative = new Tensor(sigmoid.Shape);
            Vml.Mul(sigmoid.Data.Length, sigmoid.Data, oneMinusSigmoid.Data, sigmoidDerivative.Data);

            // Finally, multiply elementwise by upstream gradient
            Tensor gradInput = new Tensor(sigmoid.Shape);
            Vml.Mul(sigmoidDerivative.Data.Length, sigmoidDerivative.Data, upstreamGradient.Data, gradInput.Data);

            return gradInput;
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise sine.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseSinReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSinReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Tensor cosTensor = new Tensor(tensorA.Shape);
            Vml.Cos(tensorA.Data.Length, tensorA.Data, cosTensor.Data);
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, cosTensor.Data, gradA.Data);

            return gradA;
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise cosine.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseCosReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseCosReverse expects exactly one initial tensor.");
            }

            Tensor tensorA = this.InitialTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Tensor sinTensor = new Tensor(tensorA.Shape);
            Vml.Sin(tensorA.Data.Length, tensorA.Data, sinTensor.Data);
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, sinTensor.Data, gradA.Data);
            Blas.scal(PradTools.NegativeOne, gradA.Data);

            return gradA;
        }

        /// <summary>
        /// Computes the gradient (reverse mode differentiation) of the Mean Squared Error (MSE) with respect to yTrue.
        /// </summary>
        /// <param name="hasBatchDimension">If true, treats the first dimension as the batch dimension and computes gradients per batch.</param>
        /// <returns>A tensor containing the gradient of the MSE with respect to yTrue.</returns>
        /// <exception cref="ArgumentException">Thrown if the shapes of the input tensors do not match.</exception>
        public Tensor MeanSquaredErrorReverse(bool hasBatchDimension = false)
        {
            Tensor yTrue = this.InitialTensors[0];
            Tensor yPred = this.InitialTensors[1];

            // Ensure the shapes of the two tensors are the same
            if (!yTrue.Shape.SequenceEqual(yPred.Shape))
            {
                throw new ArgumentException("The tensors must have the same shape for Mean Squared Error gradient calculation.");
            }

            if (hasBatchDimension && yTrue.Shape.Length > 1)
            {
                // Treat the first dimension as the batch dimension
                int batchSize = yTrue.Shape[0];
                int[] reducedShape = yTrue.Shape.Skip(1).ToArray();  // Shape for individual batches
                int numElementsPerBatch = reducedShape.Aggregate(1, (a, b) => a * b);  // Total elements in each batch

                // Preallocate a tensor to store the gradient for each batch
                var gradientData = PradTools.AllocateArray(yTrue.Data.Length);

                // Compute gradients for each batch independently
                for (int batch = 0; batch < batchSize; batch++)
                {
                    // Slice out each batch for both yTrue and yPred
                    Tensor yTrueBatch = yTrue.Slice(new int[] { batch, 0 }, reducedShape);
                    Tensor yPredBatch = yPred.Slice(new int[] { batch, 0 }, reducedShape);

                    // Step 1: Compute the element-wise difference (yTrue - yPred)
                    var difference = yTrueBatch.ElementwiseSub(yPredBatch);

                    // Step 2: Compute the gradient for each batch (2/n * (yTrue - yPred))
                    // The gradient of MSE with respect to yTrue is: 2/n * (yTrue - yPred)
                    var scaleFactor = PradTools.Two / numElementsPerBatch;
                    var scaledTensor = new Tensor(difference.Shape, scaleFactor);
                    var gradientBatch = difference.ElementwiseMultiply(scaledTensor);

                    // Step 3: Store the gradient data for the current batch
                    Array.Copy(gradientBatch.Data, 0, gradientData, batch * numElementsPerBatch, numElementsPerBatch);
                }

                // Step 4: Return the gradient as a tensor
                return new Tensor(yTrue.Shape, gradientData);
            }
            else
            {
                // No batch dimension; compute gradient over the entire tensor
                int numElements = yTrue.Data.Length;

                // Step 1: Compute the element-wise difference (yTrue - yPred)
                var difference = yTrue.ElementwiseSub(yPred);

                // Step 2: Compute the gradient for the entire tensor (2/n * (yTrue - yPred))
                var scaleFactor = PradTools.Two / numElements;
                var scaledTensor = new Tensor(difference.Shape, scaleFactor);
                var gradient = difference.ElementwiseMultiply(scaledTensor);

                // Step 3: Return the gradient as a tensor
                return gradient;
            }
        }

        /// <summary>
        /// Computes the reverse (backward) gradient for the element-wise modulus operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient of the loss with respect to the output of the modulus operation.</param>
        /// <param name="x">The original tensor `x` in the modulus operation `x % y`.</param>
        /// <returns>An array containing the gradients with respect to `x` and `y`.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the number of initial tensors is not as expected.</exception>
        public Tensor[] ModulusReverse(Tensor upstreamGradient, Tensor x)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ModulusReverse expects exactly one initial tensor.");
            }

            Tensor y = this.InitialTensors[0];

            this.CheckShapeCompatibility(y, upstreamGradient);
            this.CheckShapeCompatibility(y, x);

            var gradX = upstreamGradient.DeepClone();
            var gradY = new Tensor(y.Shape);

            // Compute floor(x / y) which is the integral part of the division
            var quotient = new Tensor(x.Shape);
            Vml.Div(x.Data, y.Data, quotient.Data);
            var integralPart = new Tensor(x.Shape);
            var fractionalPart = new Tensor(x.Shape);
            Vml.Modf(quotient.Data, integralPart.Data, fractionalPart.Data);  // integralPart = floor(x / y)

            // Compute gradY = -upstreamGradient * integralPart
            Vml.Mul(integralPart.Data.Length, integralPart.Data, upstreamGradient.Data, gradY.Data);
            Blas.scal(PradTools.NegativeOne, gradY.Data);

            return new Tensor[] { gradX, gradY };
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise atan2 operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="x">The other tensor used in the atan2 operation.</param>
        /// <returns>The gradients with respect to the input tensors.</returns>
        /// <exception cref="ArgumentException">If the shapes of the tensors are not compatible.</exception>
        public Tensor[] ElementwiseAtan2Reverse(Tensor upstreamGradient, Tensor x)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseAtan2Reverse expects exactly one initial tensor.");
            }

            Tensor y = this.InitialTensors[0];

            this.CheckShapeCompatibility(y, upstreamGradient);
            this.CheckShapeCompatibility(y, x);

            var gradY = new Tensor(y.Shape);
            var gradX = new Tensor(x.Shape);

            var ySquared = new Tensor(y.Shape);
            var xSquared = new Tensor(x.Shape);
            var denominator = new Tensor(y.Shape);

            // Compute y^2 and x^2
            Vml.Pow(y.Data.Length, y.Data, Enumerable.Repeat(PradTools.Two, y.Data.Length).ToArray(), ySquared.Data);
            Vml.Pow(x.Data.Length, x.Data, Enumerable.Repeat(PradTools.Two, x.Data.Length).ToArray(), xSquared.Data);

            // Compute denominator = y^2 + x^2
            Vml.Add(ySquared.Data.Length, ySquared.Data, xSquared.Data, denominator.Data);

            var epsilon = PradTools.Epsilon10;
            var normalizedDenominator = new Tensor(x.Shape);
            var epsilonTensor = new Tensor(x.Shape, epsilon);

            Vml.MaxMag(denominator.Data.Length, denominator.Data, epsilonTensor.Data, normalizedDenominator.Data);

            // Compute gradY = upstreamGradient * x / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, x.Data, gradY.Data);
            Vml.Div(gradY.Data.Length, gradY.Data, normalizedDenominator.Data, gradY.Data);

            // Compute gradX = -upstreamGradient * y / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, y.Data, gradX.Data);
            Vml.Div(gradX.Data.Length, gradX.Data, normalizedDenominator.Data, gradX.Data);
            Blas.scal(PradTools.NegativeOne, gradX.Data);

            return new Tensor[] { gradY, gradX };
        }

        /// <summary>
        /// Computes the reverse gradient for CreateFlatArray.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="indices">The indices used in CreateFlatArray.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] CreateFlatArrayReverse(Tensor upstreamGradient, int[] indices)
        {
            int numTensors = this.InitialTensors.Length;
            int[] shape = this.InitialTensors[0].Shape;

            // Initialize the gradient tensors for each input tensor
            Tensor[] gradients = new Tensor[numTensors];
            for (int i = 0; i < numTensors; i++)
            {
                gradients[i] = new Tensor(shape);
            }

            int flatIndex = 0;

            for (int t = 0; t < numTensors; t++)
            {
                Tensor tensor = this.InitialTensors[t];

                foreach (var index in indices)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        gradients[t][i, index] += upstreamGradient.Data[flatIndex++];
                    }
                }
            }

            return gradients;
        }

        /// <summary>
        /// The reverse stack operation.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The gradients.</returns>
        /// <exception cref="InvalidOperationException">Empty input list.</exception>
        /// <exception cref="ArgumentException">Out of bounds axis.</exception>
        public Tensor[] StackReverse(Tensor upstreamGradient, int axis = 0)
        {
            if (this.InitialTensors == null || this.InitialTensors.Length == 0)
            {
                throw new InvalidOperationException("The input list of initial tensors cannot be empty.");
            }

            int numTensors = this.InitialTensors.Length;
            int[] gradShape = upstreamGradient.Shape;

            // Handle negative axis
            if (axis < 0)
            {
                axis = gradShape.Length + axis;
            }

            // Validate axis
            if (axis < 0 || axis >= gradShape.Length)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Calculate the shape of individual gradient tensors
            int[] tensorShape = new int[gradShape.Length - 1];
            Array.Copy(gradShape, 0, tensorShape, 0, axis);
            Array.Copy(gradShape, axis + 1, tensorShape, axis, gradShape.Length - axis - 1);

            // Initialize gradient tensors
            var gradients = new Tensor[numTensors];
            for (int i = 0; i < numTensors; i++)
            {
                gradients[i] = new Tensor(tensorShape);
            }

            // Calculate the size of each individual tensor
            int tensorSize = tensorShape.Aggregate(1, (a, b) => a * b);

            // Distribute the gradient
            Parallel.For(0, numTensors, i =>
            {
                int start = i * tensorSize;
                Array.Copy(upstreamGradient.Data, start, gradients[i].Data, 0, tensorSize);
            });

            return gradients;
        }

        /// <summary>
        /// Computes the reverse gradient for the Squeeze operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="originalShape">The original shape of the tensor before squeeze.</param>
        /// <param name="axes">The axes that were squeezed. If null, all axes of size 1 were removed.</param>
        /// <returns>The gradient with respect to the input tensor in its original shape.</returns>
        public Tensor SqueezeReverse(Tensor upstreamGradient, int[] originalShape, int[]? axes = null)
        {
            if (axes == null)
            {
                axes = Enumerable.Range(0, originalShape.Length).Where(i => originalShape[i] == 1).ToArray();
            }

            int[] newShape = upstreamGradient.Shape;
            int[] expandedShape = new int[originalShape.Length];

            int j = 0;
            for (int i = 0; i < originalShape.Length; i++)
            {
                if (axes.Contains(i))
                {
                    expandedShape[i] = 1;
                }
                else
                {
                    expandedShape[i] = newShape[j++];
                }
            }

            return upstreamGradient.Reshape(expandedShape);
        }

        /// <summary>
        /// Computes the gradient for the reciprocal operation using MKL.NET.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing back from the subsequent operation.</param>
        /// <param name="reciprocalResult">The result of the forward reciprocal operation.</param>
        /// <returns>The computed gradient.</returns>
        public Tensor ReciprocalReverse(Tensor upstreamGradient, Tensor reciprocalResult)
        {
            Tensor gradient = new Tensor(upstreamGradient.Shape);

            // Compute -upstreamGradient * reciprocalResult^2
            Tensor squaredReciprocal = new Tensor(reciprocalResult.Shape);
            Vml.Sqr(reciprocalResult.Data.Length, reciprocalResult.Data, squaredReciprocal.Data);

            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, squaredReciprocal.Data, gradient.Data);

            Tensor negativeOnes = new Tensor(gradient.Shape, PradTools.NegativeOne);

            Vml.Mul(gradient.Data.Length, gradient.Data, negativeOnes.Data, gradient.Data);

            return gradient;
        }

        /// <summary>
        /// Computes the gradient of the Upsample operation with respect to the input tensor during the backward pass.
        /// Accumulates gradients from the output to the corresponding input positions.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layers with the same shape as the upsampled output.</param>
        /// <param name="scaleFactor">The scaling factor used in the forward pass for upsampling.</param>
        /// <param name="method">The method used in forward pass ("nearest" or "bilinear").</param>
        /// <returns>A new tensor representing the gradient with respect to the input tensor.</returns>
        public Tensor UpsampleReverse(Tensor upstreamGradient, int scaleFactor, string method = "nearest")
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseAtan2Reverse expects exactly one initial tensor.");
            }

            Tensor input = this.InitialTensors[0];
            int[] inputShape = input.Shape;
            int batchSize, channels, inputHeight, inputWidth;

            if (inputShape.Length == 2)
            {
                batchSize = 1;
                channels = 1;
                inputHeight = inputShape[0];
                inputWidth = inputShape[1];
            }
            else if (inputShape.Length == 3)
            {
                batchSize = 1;
                inputHeight = inputShape[0];
                inputWidth = inputShape[1];
                channels = inputShape[2];
            }
            else if (inputShape.Length == 4)
            {
                batchSize = inputShape[0];
                inputHeight = inputShape[1];
                inputWidth = inputShape[2];
                channels = inputShape[3];
            }
            else
            {
                throw new ArgumentException("Upsampling only supports 2D or 4D tensors.");
            }

            // Create a tensor to accumulate the gradients for the input
            var inputGradient = new Tensor(input.Shape);

            if (method == "nearest")
            {
                // Backward pass for nearest-neighbor upsampling
                Parallel.For(0, batchSize, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < upstreamGradient.Shape[1]; i++)
                        {
                            int nearestY = i / scaleFactor; // Corresponding input Y

                            for (int j = 0; j < upstreamGradient.Shape[2]; j++)
                            {
                                int nearestX = j / scaleFactor; // Corresponding input X

                                // Accumulate the gradients from the output back to the input
                                inputGradient[b, nearestY, nearestX, c] += upstreamGradient[b, i, j, c];
                            }
                        }
                    }
                });
            }
            else if (method == "bilinear")
            {
                // Backward pass for bilinear interpolation
                Parallel.For(0, batchSize, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < upstreamGradient.Shape[1]; i++)
                        {
                            float srcY = (float)i / scaleFactor;
                            int y0 = (int)Math.Floor(srcY);
                            int y1 = Math.Min(y0 + 1, inputHeight - 1);
                            float dy = srcY - y0;

                            for (int j = 0; j < upstreamGradient.Shape[2]; j++)
                            {
                                float srcX = (float)j / scaleFactor;
                                int x0 = (int)Math.Floor(srcX);
                                int x1 = Math.Min(x0 + 1, inputWidth - 1);
                                float dx = srcX - x0;

                                // Bilinear interpolation gradient contribution
                                var grad = upstreamGradient[b, i, j, c];

                                // Gradient with respect to top-left
                                inputGradient[b, y0, x0, c] += (1 - dx) * (1 - dy) * grad;

                                // Gradient with respect to top-right
                                inputGradient[b, y0, x1, c] += dx * (1 - dy) * grad;

                                // Gradient with respect to bottom-left
                                inputGradient[b, y1, x0, c] += (1 - dx) * dy * grad;

                                // Gradient with respect to bottom-right
                                inputGradient[b, y1, x1, c] += dx * dy * grad;
                            }
                        }
                    }
                });
            }
            else
            {
                throw new ArgumentException($"Unsupported method: {method}. Use 'nearest' or 'bilinear'.");
            }

            return inputGradient;
        }

        /// <summary>
        /// Reverses the interleaved gather operation by using the upstream gradient tensor to map back gradients
        /// to their original positions in the original tensor.
        /// </summary>
        /// <param name="upstreamGradient">The gradient tensor passed from the previous layer.</param>
        /// <param name="skip">The number of row indices skipped during interleaving.</param>
        /// <param name="restart">The number of rows before restarting the pattern.</param>
        /// <returns>The gradient of the original tensor, reshaped to its original form.</returns>
        public Tensor InterleavedGatherReverse(Tensor upstreamGradient, int skip, int restart)
        {
            // Step 1: Reshape the upstream gradient tensor from [1, 3, 3, 6] to [9, 6] (flatten matrix)
            int totalRowsToGather = (upstreamGradient.Shape[^1] / 2) * upstreamGradient.Shape[1];
            int[] reshapedShape = new int[] { upstreamGradient.Shape[1] * upstreamGradient.Shape[2], upstreamGradient.Shape[^1] }; // Flatten to [9, 6]
            Tensor reshapedTensor = upstreamGradient.Reshape(reshapedShape); // Reshape to [9, 6]

            // Step 2: Transpose from [9, 6] to [6, 9] to separate components
            Tensor transposedTensor = reshapedTensor.Transpose(1, 0); // Transpose to [6, 9]

            // Step 3: Prepare result tensor where we will gather the interleaved rows back to original positions
            Tensor resultTensor = new Tensor(reshapedShape); // Prepare result tensor with shape [9, 6]

            int resultIndex = 0; // Index for storing values in result tensor

            // Step 4: Perform the same interleaving reverse logic using the upstream gradient values
            for (int iteration = 0; iteration < upstreamGradient.Shape[1]; iteration++)
            {
                // Step 4.1: Copy magnitudes for the current channel from upstream gradient
                Array.Copy(transposedTensor.Data, iteration * totalRowsToGather, resultTensor.Data, resultIndex * restart, skip);    // First set of magnitudes
                Array.Copy(transposedTensor.Data, (iteration * totalRowsToGather) + skip, resultTensor.Data, (resultIndex + 1) * restart, skip); // Second set
                Array.Copy(transposedTensor.Data, (iteration * totalRowsToGather) + (skip * 2), resultTensor.Data, (resultIndex + 2) * restart, skip); // Third set

                // Step 4.2: Copy angles for the current channel from upstream gradient
                Array.Copy(transposedTensor.Data, (iteration + skip) * totalRowsToGather, resultTensor.Data, (resultIndex * restart) + skip, skip);  // First set of angles
                Array.Copy(transposedTensor.Data, ((iteration + skip) * totalRowsToGather) + skip, resultTensor.Data, ((resultIndex + 1) * restart) + skip, skip); // Second set
                Array.Copy(transposedTensor.Data, ((iteration + skip) * totalRowsToGather) + (skip * 2), resultTensor.Data, ((resultIndex + 2) * restart) + skip, skip); // Third set

                // Step 4.3: Move to the next block of rows for the next iteration (green, blue channels, etc.)
                resultIndex += 3;
            }

            // Step 5: Reshape the result back to the original shape [3, 3, 6]
            return resultTensor.Reshape(new int[] { (upstreamGradient.Shape[^1] / 2), upstreamGradient.Shape[1], upstreamGradient.Shape[^2] * 2 });
        }

        /// <summary>
        /// Performs an interleaved gather operation on the upstream gradient tensor with efficient copying and correct looping logic.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient tensor.</param>
        /// <param name="skip">The number of row indices to skip during the interleaving process.</param>
        /// <param name="restart">The number of rows before restarting the pattern.</param>
        /// <returns>A new tensor with interleaved gathered results.</returns>
        public Tensor InterleavedGatherInverseReverse(Tensor upstreamGradient, int skip, int restart)
        {
            // Step 1: Reshape the upstream gradient tensor to [3, 18] (assuming original shape [3, 3, 6])
            int[] reshapedShape = new int[] { upstreamGradient.Shape[0], upstreamGradient.Shape[1] * upstreamGradient.Shape[2] };
            Tensor reshapedTensor = upstreamGradient.Reshape(reshapedShape);

            // Step 2: Transpose the reshaped tensor to [18, 3] for interleaving
            Tensor transposedTensor = reshapedTensor.Transpose(1, 0); // Transpose to [18, 3]

            // Step 3: Prepare result tensor with shape [number of interleaved rows, batch size (3)]
            int totalRowsToGather = transposedTensor.Shape[0];  // You are gathering all rows from the transposed tensor
            Tensor resultTensor = new Tensor(new int[] { totalRowsToGather, transposedTensor.Shape[1] }); // Shape [18, 3]

            int resultIndex = 0;  // Index for storing values in the result tensor

            // Single loop for i++
            for (int i = 0; i < transposedTensor.Shape[0]; i++)
            {
                // Copy the row at index i from the transposed tensor into the result tensor
                Array.Copy(transposedTensor.Data, i * upstreamGradient.Shape[0], resultTensor.Data, resultIndex * upstreamGradient.Shape[0], upstreamGradient.Shape[0]);
                resultIndex++;

                // Apply the interleaved skip and copy the next set of data
                Array.Copy(transposedTensor.Data, (i + skip) * upstreamGradient.Shape[0], resultTensor.Data, resultIndex * upstreamGradient.Shape[0], upstreamGradient.Shape[0]);
                resultIndex++;

                // Only reset i when we have completed the restart block
                if (resultIndex % restart == 0)
                {
                    i += skip;  // Move i to the next block for the next iteration
                }
            }

            // Step 4: Reshape the result back to the original shape (e.g., [1, 3, 3, 6])
            return resultTensor.Reshape(new int[] { 1, upstreamGradient.Shape[1], skip, upstreamGradient.Shape[0] * 2 });
        }

        /// <summary>
        /// Computes the reverse gradient for the Gather operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="indices">The indices of elements that were gathered.</param>
        /// <param name="axis">The axis along which slices were gathered.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor GatherReverse(Tensor upstreamGradient, Tensor indices, int axis = 0)
        {
            Tensor inputTensor = this.InitialTensors[0];
            int[] inputShape = inputTensor.Shape;
            Tensor grad = new Tensor(inputShape);

            if (axis < 0 || axis >= inputShape.Length)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Handle negative indices and validate
            var indicesData = indices.Data.Select(i => i < 0 ? inputShape[axis] + (int)i : (int)i).ToArray();
            foreach (var index in indicesData)
            {
                if (index < 0 || index >= inputShape[axis])
                {
                    throw new ArgumentException("Index out of bounds.");
                }
            }

            // Calculate strides for the input tensor
            int[] strides = new int[inputShape.Length];
            strides[inputShape.Length - 1] = 1;
            for (int i = inputShape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * inputShape[i + 1];
            }

            // Calculate the shape of the upstream gradient
            int[] upstreamShape = new int[inputShape.Length + indices.Shape.Length - 1];
            Array.Copy(inputShape, 0, upstreamShape, 0, axis);
            Array.Copy(indices.Shape, 0, upstreamShape, axis, indices.Shape.Length);
            Array.Copy(inputShape, axis + 1, upstreamShape, axis + indices.Shape.Length, inputShape.Length - axis - 1);

            Parallel.For(0, upstreamGradient.Data.Length, i =>
            {
                int[] upstreamIndices = new int[upstreamShape.Length];
                int temp = i;
                for (int j = upstreamShape.Length - 1; j >= 0; j--)
                {
                    upstreamIndices[j] = temp % upstreamShape[j];
                    temp /= upstreamShape[j];
                }

                int gradIndex = 0;
                int upstreamIdx = 0;
                for (int j = 0; j < inputShape.Length; j++)
                {
                    if (j == axis)
                    {
                        gradIndex += indicesData[upstreamIndices[upstreamIdx]] * strides[j];
                        upstreamIdx++;
                    }
                    else
                    {
                        gradIndex += upstreamIndices[upstreamIdx] * strides[j];
                        upstreamIdx++;
                    }
                }

                lock (grad.Data)
                {
                    grad.Data[gradIndex] += upstreamGradient.Data[i];
                }
            });

            return grad;
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise subtraction.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseSubReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseSubReverse expects exactly two initial tensors.");
            }

            Tensor tensorA = this.InitialTensors[0];
            Tensor tensorB = this.InitialTensors[1];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorB, upstreamGradient);

            // Gradient with respect to A is just the upstream gradient
            Tensor gradA = upstreamGradient;

            // Gradient with respect to B is the negative of the upstream gradient
            Tensor gradB = new Tensor(upstreamGradient.Shape);
            Blas.copy(upstreamGradient.Data, gradB.Data);
            Blas.scal(PradTools.NegativeOne, gradB.Data);

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the gradient for the embeddings tensor based on the upstream gradient tensor.
        /// This method implements the reverse mode of the embedding operation.
        /// </summary>
        /// <param name="upstream">The upstream gradient tensor with respect to the output of the embedding operation.</param>
        /// <returns>The gradient for the embeddings tensor.</returns>
        /// <exception cref="ArgumentException">Thrown when tensor shapes are invalid.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when an index is out of range.</exception>
        public Tensor EmbeddingReverse(Tensor upstream)
        {
            var indices = this.InitialTensors[0];
            var embeddings = this.InitialTensors[1];

            // Validate the upstream gradient tensor
            if (upstream.Shape.Length < 2 || upstream.Shape.Length > 4)
            {
                throw new ArgumentException("The upstream gradient tensor must be 2D, 3D, or 4D.");
            }

            // Validate the embeddings tensor
            if (embeddings.Shape.Length != 2 && embeddings.Shape.Length != 3)
            {
                throw new ArgumentException("The embeddings tensor must be 2D or 3D.");
            }

            // Check if embeddings are batched
            bool batchedEmbeddings = embeddings.Shape.Length == 3;

            // Determine the number of embeddings and embedding dimension
            int numEmbeddings = batchedEmbeddings ? embeddings.Shape[1] : embeddings.Shape[0];
            int embeddingDim = batchedEmbeddings ? embeddings.Shape[2] : embeddings.Shape[1];

            // Initialize the gradient tensor for embeddings (same shape as embeddings tensor)
            var embeddingGrad = new Tensor(embeddings.Shape);

            // Create a thread-local copy of the gradients
            var localGradients = new Tensor[Environment.ProcessorCount];
            for (int t = 0; t < localGradients.Length; t++)
            {
                localGradients[t] = new Tensor(embeddings.Shape);
            }

            // Perform the gradient accumulation
            Parallel.For(0, indices.Data.Length, i =>
            {
                int threadId = Thread.CurrentThread.ManagedThreadId % Environment.ProcessorCount;

                int index = (int)indices.Data[i];

                // Compute the batch index for batched embeddings
                int batchIndex = batchedEmbeddings ? i / (indices.Shape[^2] * indices.Shape[^1]) : 0;

                // Calculate the source offset in the upstream gradient tensor
                int srcOffset = i * embeddingDim;

                // Calculate the destination offset in the local embedding gradient tensor
                int destOffset = batchedEmbeddings
                    ? ((batchIndex * numEmbeddings) + index) * embeddingDim
                    : index * embeddingDim;

                // Accumulate the gradients into the thread-local buffer
                for (int j = 0; j < embeddingDim; j++)
                {
                    localGradients[threadId].Data[destOffset + j] += upstream.Data[srcOffset + j];
                }
            });

            // Sum up the local gradients into the final gradient tensor
            for (int t = 0; t < localGradients.Length; t++)
            {
                for (int i = 0; i < embeddingGrad.Data.Length; i++)
                {
                    embeddingGrad.Data[i] += localGradients[t].Data[i];
                }
            }

            return embeddingGrad;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise division operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="other">The other tensor used in the division operation.</param>
        /// <returns>The gradients with respect to the input tensors.</returns>
        /// <exception cref="ArgumentException">If the shapes of the tensors are not compatible.</exception>
        public Tensor[] ElementwiseDivideReverse(Tensor upstreamGradient, Tensor other)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseDivideReverse expects exactly one initial tensor.");
            }

            Tensor a = this.InitialTensors[0];
            Tensor b = other;

            this.CheckShapeCompatibility(a, upstreamGradient);
            this.CheckShapeCompatibility(a, b);

            var gradA = new Tensor(a.Shape);
            var gradB = new Tensor(b.Shape);

            // Compute gradA = upstreamGradient / b
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, b.Data, gradA.Data);

            // Compute gradB = -upstreamGradient * a / (b * b)
            var bSquared = new Tensor(b.Shape);
            Vml.Mul(b.Data.Length, b.Data, b.Data, bSquared.Data);

            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, a.Data, gradB.Data);
            Vml.Div(gradB.Data.Length, gradB.Data, bSquared.Data, gradB.Data);
            Blas.scal(PradTools.NegativeOne, gradB.Data);

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes and returns the reverse-mode gradient for differentiation along the specified axis using SIMD acceleration.
        /// </summary>
        /// <param name="gradOutputTensor">Upstream gradient from autodiff system.</param>
        /// <param name="axis">Axis along which the forward diff was computed (0 for vertical, 1 for horizontal).</param>
        /// <returns>The computed gradient tensor.</returns>
        public Tensor DiffReverse(Tensor gradOutputTensor, int axis)
        {
            if (gradOutputTensor == null)
            {
                throw new ArgumentNullException(nameof(gradOutputTensor));
            }

            var shape = gradOutputTensor.Shape;
            if (shape.Length != 2)
            {
                throw new ArgumentException("DiffReverse requires 2D tensors");
            }

            int height = shape[0];
            int width = shape[1];

            var gradOutput = gradOutputTensor.Data;
            var gradInput = PradTools.AllocateArray(gradOutput.Length);

            Tensor gradInputT = new Tensor(shape, gradInput);
            switch (axis)
            {
                case 0:
                    this.ReverseDiffVerticalSIMD(gradInputT, new Tensor(shape, gradOutput), height, width);
                    break;
                case 1:
                    this.ReverseDiffHorizontalSIMD(gradInputT, new Tensor(shape, gradOutput), height, width);
                    break;
                default:
                    throw new ArgumentException("Axis must be 0 or 1", nameof(axis));
            }

            return new Tensor(shape, gradInput);
        }

        /// <summary>
        /// Computes the reverse gradient for the ExpandDims operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ExpandDimsReverse(Tensor upstreamGradient, int axis = -1)
        {
            // Normalize the axis in case it's negative
            if (axis < 0)
            {
                axis += upstreamGradient.Shape.Length;
            }

            // Ensure the axis is valid
            if (axis < 0 || axis >= upstreamGradient.Shape.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(axis), "Axis is out of range.");
            }

            // Remove the expanded dimension
            int[] newShape = new int[upstreamGradient.Shape.Length - 1];
            for (int i = 0; i < axis; i++)
            {
                newShape[i] = upstreamGradient.Shape[i];
            }

            for (int i = axis + 1; i < upstreamGradient.Shape.Length; i++)
            {
                newShape[i - 1] = upstreamGradient.Shape[i];
            }

            // Sum along the expanded dimension
            return upstreamGradient.Reshape(newShape);
        }

        /// <summary>
        /// Computes the gradient tensors for the inputs of the PairwiseTile function.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient tensor of shape [2, N * P].</param>
        /// <returns>A tuple of two gradient tensors corresponding to the inputs of PairwiseTile.</returns>
        public (Tensor gradTensor1, Tensor gradTensor2) PairwiseTileReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("TileReverse expects exactly two initial tensors.");
            }

            Tensor tensor1 = this.InitialTensors[0];
            Tensor tensor2 = this.InitialTensors[1];

            int n = tensor1.Shape[1];
            int p = tensor2.Shape[1];

            // Create gradient tensors for tensor1 and tensor2
            var gradTensor1 = new Tensor(new int[] { 1, n });
            var gradTensor2 = new Tensor(new int[] { 1, p });

            // Validate the shape of the upstream gradient
            if (upstreamGradient.Shape.Length != 2 || upstreamGradient.Shape[0] != 2 || upstreamGradient.Shape[1] != n * p)
            {
                throw new ArgumentException("Upstream gradient must be of shape [2, N * P].");
            }

            // Calculate the gradient for tensor1
            Parallel.For(0, n, i =>
            {
                // Sum the relevant part of the upstream gradient for tensor1
                var sum = PradTools.Zero;
                for (int j = 0; j < p; j++)
                {
                    sum += upstreamGradient.Data[(i * p) + j];  // Accumulate values for the first row
                }

                gradTensor1.Data[i] = sum;
            });

            // Calculate the gradient for tensor2
            Parallel.For(0, p, j =>
            {
                // Sum the relevant part of the upstream gradient for tensor2
                var sum = PradTools.Zero;
                for (int i = 0; i < n; i++)
                {
                    sum += upstreamGradient.Data[(n * p) + ((i * p) + j)];  // Accumulate values for the second row
                }

                gradTensor2.Data[j] = sum;
            });

            return (gradTensor1, gradTensor2);
        }

        /// <summary>
        /// Computes the reverse gradient for the Tile operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor TileReverse(Tensor upstreamGradient, int[] multiples)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("TileReverse expects exactly one initial tensor.");
            }

            Tensor originalTensor = this.InitialTensors[0];
            if (multiples.Length != originalTensor.Shape.Length)
            {
                throw new ArgumentException("Length of multiples must match the number of dimensions of the tensor.");
            }

            if (multiples.Length == 4 && multiples[0] == 1 && multiples[1] == 1 && multiples[3] == 1)
            {
                var res = this.TileReverseSpecializedCaseSIMD(upstreamGradient, originalTensor.Shape, multiples[2]);
                return res;
            }
            else if (multiples.Length == 4 && multiples[0] == 1 && multiples[2] == 1 && multiples[3] == 1)
            {
                var res = this.TileReverseSpecializedCaseSecondDim(upstreamGradient, originalTensor.Shape, multiples[1]);
                return res;
            }
            else if (multiples.Length == 2 && multiples[1] == 1)
            {
                var res = this.TileReverseSpecializedCaseN1SIMD(upstreamGradient, originalTensor.Shape, multiples[0]);
                return res;
            }
            else if (multiples.Length == 2 && multiples[0] == 1)
            {
                var res = this.TileReverseSpecializedCase1NSIMD(upstreamGradient, originalTensor.Shape, multiples[1]);
                return res;
            }

            int[] originalShape = originalTensor.Shape;
            int[] tiledShape = new int[originalShape.Length];
            for (int i = 0; i < originalShape.Length; i++)
            {
                tiledShape[i] = originalShape[i] * multiples[i];
            }

            if (!upstreamGradient.Shape.SequenceEqual(tiledShape))
            {
                throw new ArgumentException("Upstream gradient shape does not match the tiled tensor shape.");
            }

            Tensor grad = new Tensor(originalShape);

            if (multiples.All(m => m == 1))
            {
                Array.Copy(upstreamGradient.Data, grad.Data, upstreamGradient.Data.Length);
                return grad;
            }

            // Iterate over all elements in the upstream gradient
            for (int i = 0; i < upstreamGradient.Data.Length; i++)
            {
                int[] tiledIndices = this.GetMultiDimensionalIndices(i, tiledShape);
                int[] originalIndices = new int[originalShape.Length];

                // Calculate corresponding indices in the original tensor
                for (int j = 0; j < originalShape.Length; j++)
                {
                    originalIndices[j] = tiledIndices[j] % originalShape[j];
                }

                // Accumulate gradient
                grad[originalIndices] += upstreamGradient.Data[i];
            }

            return grad;
        }

        /// <summary>
        /// Computes the reverse gradient for GatherNd.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="indices">The tensor containing the indices.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor GatherNdReverse(Tensor upstreamGradient, Tensor indices)
        {
            Tensor inputTensor = this.InitialTensors[0];

            if (indices.Shape[^1] != inputTensor.Shape.Length)
            {
                throw new ArgumentException("The last dimension of indices must match the rank of the input tensor.");
            }

            int[] resultShape = inputTensor.Shape;
            var result = new Tensor(resultShape);

            int[] inputShape = inputTensor.Shape;
            int[] indicesShape = indices.Shape;
            int lastDim = indicesShape[^1];
            int numSlices = indicesShape.Take(indicesShape.Length - 1).Aggregate(1, (a, b) => a * b);

            Parallel.For(0, numSlices, i =>
            {
                int[] index = this.GetMultiDimensionalIndices(i, indicesShape.Take(indicesShape.Length - 1).ToArray());
                int[] sourceIndex = new int[lastDim];

                for (int j = 0; j < lastDim; j++)
                {
                    sourceIndex[j] = (int)indices[index.Concat(new int[] { j }).ToArray()];
                }

                int flatIndex = this.GetIndex(sourceIndex, inputShape);
                lock (result.Data)
                {
                    result.Data[flatIndex] += upstreamGradient.Data[i];
                }
            });

            return result;
        }

        /// <summary>
        /// Computes the reverse gradient for the reshape operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="originalShape">The original shape of the tensor before reshape.</param>
        /// <returns>The gradient with respect to the input tensor in its original shape.</returns>
        public Tensor ReshapeReverse(Tensor upstreamGradient, int[] originalShape)
        {
            int originalTotalSize = this.GetTotalSize(originalShape);
            int upstreamTotalSize = upstreamGradient.Data.Length;

            if (originalTotalSize != upstreamTotalSize)
            {
                throw new ArgumentException("The total number of elements must match between the original shape and the upstream gradient.");
            }

            var reshapedGradient = new Tensor(originalShape, upstreamGradient.Data);
            return reshapedGradient;
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
            Tensor inputTensor = this.InitialTensors[0];

            if (begin.Length != inputTensor.Shape.Length || size.Length != inputTensor.Shape.Length || (strides != null && strides.Length != inputTensor.Shape.Length))
            {
                throw new ArgumentException("The lengths of begin, size, and strides must match the number of dimensions of the tensor.");
            }

            strides = strides ?? Enumerable.Repeat(1, inputTensor.Shape.Length).ToArray();
            for (int i = 0; i < inputTensor.Shape.Length; i++)
            {
                if (strides[i] == 0)
                {
                    throw new ArgumentException("Stride cannot be zero.");
                }
            }

            // Adjust begin indices for negative values and calculate the effective size if size is negative
            int[] adjustedBegin = (int[])begin.Clone();
            int[] effectiveSize = (int[])size.Clone();
            for (int i = 0; i < inputTensor.Shape.Length; i++)
            {
                if (adjustedBegin[i] < 0)
                {
                    adjustedBegin[i] += inputTensor.Shape[i];
                }

                if (effectiveSize[i] < 0)
                {
                    effectiveSize[i] = (inputTensor.Shape[i] - adjustedBegin[i]) / Math.Abs(strides[i]);
                }

                if (adjustedBegin[i] < 0 || adjustedBegin[i] >= inputTensor.Shape[i] ||
                    (adjustedBegin[i] + ((effectiveSize[i] - 1) * strides[i]) < 0 || adjustedBegin[i] + ((effectiveSize[i] - 1) * strides[i]) >= inputTensor.Shape[i]))
                {
                    throw new ArgumentException("The slice extends beyond the boundaries of the tensor.");
                }
            }

            // Create a tensor for the gradient of the original input
            var inputGradient = new Tensor(inputTensor.Shape);

            // Reverse the slicing operation by distributing the upstream gradient back to the appropriate positions in the input gradient
            Parallel.For(0, upstreamGradient.Data.Length, upstreamIndex =>
            {
                int[] resultIndices = upstreamGradient.GetMultiDimensionalIndices(upstreamIndex, upstreamGradient.Shape);
                int[] sourceIndices = new int[resultIndices.Length];
                for (int i = 0; i < resultIndices.Length; i++)
                {
                    sourceIndices[i] = adjustedBegin[i] + (resultIndices[i] * strides[i]);
                }

                // Ensure the source indices are within the bounds of the input tensor
                bool withinBounds = true;
                for (int i = 0; i < sourceIndices.Length; i++)
                {
                    if (sourceIndices[i] < 0 || sourceIndices[i] >= inputTensor.Shape[i])
                    {
                        withinBounds = false;
                        break;
                    }
                }

                if (withinBounds)
                {
                    inputGradient[sourceIndices] += upstreamGradient[resultIndices];
                }
            });

            return inputGradient;
        }

        /// <summary>
        /// The reverse of the sum rows operation.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <returns>The resultant tensor.</returns>
        /// <exception cref="ArgumentException">Shape does not match.</exception>
        public Tensor SumRowsReverse(Tensor upstreamGradient)
        {
            Tensor initialTensor = this.InitialTensors[0];
            int[] originalShape = initialTensor.Shape;

            if (upstreamGradient.Shape.Length != 2 || upstreamGradient.Shape[0] != originalShape[0] || upstreamGradient.Shape[1] != 1)
            {
                throw new ArgumentException("Upstream gradient shape does not match the expected shape.");
            }

            // Create the gradient tensor with the same shape as the original tensor
            Tensor grad = new Tensor(originalShape);

            // Distribute the upstream gradient across the columns of each row
            Parallel.For(0, originalShape[0], i =>
            {
                for (int j = 0; j < originalShape[1]; j++)
                {
                    grad[i, j] = upstreamGradient[i, 0];
                }
            });

            return grad;
        }

        /// <summary>
        /// Computes the reverse gradient for the transpose operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="permutation">The permutation of the axes used in the forward transpose operation.</param>
        /// <returns>The gradient with respect to the input tensor before transposition.</returns>
        public Tensor TransposeReverse(Tensor upstreamGradient, int[] permutation)
        {
            if (this.InitialTensors.Length != 1)
            {
                throw new InvalidOperationException("TransposeReverse expects exactly one initial tensor.");
            }

            Tensor originalTensor = this.InitialTensors[0];

            if (permutation.Length != originalTensor.Shape.Length)
            {
                throw new ArgumentException("The permutation must have the same length as the number of dimensions of the tensor.");
            }

            if (permutation.Distinct().Count() != permutation.Length || permutation.Any(p => p < 0 || p >= originalTensor.Shape.Length))
            {
                throw new ArgumentException("The permutation must be a valid permutation of the dimensions.");
            }

            // Calculate the inverse permutation
            int[] inversePermutation = new int[permutation.Length];
            for (int i = 0; i < permutation.Length; i++)
            {
                inversePermutation[permutation[i]] = i;
            }

            // The shape of the result tensor should be the same as the original tensor
            var result = new Tensor(originalTensor.Shape);

            if (originalTensor.Shape.Length == 2 && permutation.SequenceEqual(new int[] { 1, 0 }))
            {
                // 2D case
                int rows = originalTensor.Shape[0]; // Original dimensions
                int cols = originalTensor.Shape[1];

                Blas.omatcopy(LayoutChar.RowMajor, TransChar.Yes, cols, rows, PradTools.One, upstreamGradient.Data, rows, result.Data, cols);
            }
            else if (originalTensor.Shape.Length == 3 && permutation.SequenceEqual(new int[] { 0, 2, 1 }))
            {
                // 3D batch case
                int batchSize = originalTensor.Shape[0];
                int rows = originalTensor.Shape[1];  // Original dimensions
                int cols = originalTensor.Shape[2];
                int sliceSize = rows * cols;
                var resultSlice = PradTools.AllocateArray(sliceSize);

                for (int b = 0; b < batchSize; b++)
                {
                    var upstreamSlice = upstreamGradient.Data.AsSpan(b * sliceSize, sliceSize);

                    Blas.omatcopy(
                        LayoutChar.RowMajor,
                        TransChar.Yes,
                        cols,                 // Number of rows in upstreamGradient slice
                        rows,                 // Number of columns in upstreamGradient slice
                        PradTools.One,                  // Alpha (scale factor)
                        upstreamSlice.ToArray(),
                        rows,                 // Leading dimension of upstreamGradient slice
                        resultSlice,
                        cols);

                    Buffer.BlockCopy(resultSlice, 0, result.Data, b * sliceSize * PradTools.SizeOf, sliceSize * PradTools.SizeOf);
                }
            }
            else
            {
                // General case for higher dimensions
                // Perform the reverse transposition
                Parallel.For(0, result.Data.Length, i =>
                {
                    int[] originalIndices = originalTensor.GetMultiDimensionalIndices(i, originalTensor.Shape);
                    int[] transposedIndices = new int[originalIndices.Length];
                    for (int j = 0; j < originalIndices.Length; j++)
                    {
                        transposedIndices[inversePermutation[j]] = originalIndices[j];
                    }

                    int transposedIndex = upstreamGradient.GetIndex(transposedIndices);
                    result.Data[i] = upstreamGradient.Data[transposedIndex];
                });
            }

            return result;
        }

        /// <summary>
        /// Computes the reverse gradient for the Indexer operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="indices">The indices used to slice. Null values select the entire dimension.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor IndexerReverse(Tensor upstreamGradient, params string?[] indices)
        {
            Tensor inputTensor = this.InitialTensors[0];

            // Handle '...' (ellipsis) by expanding it to select all preceding dimensions
            indices = this.ExpandEllipsis(indices, inputTensor.Shape.Length);

            if (indices.Length != inputTensor.Shape.Length)
            {
                throw new ArgumentException($"Number of indices ({indices.Length}) does not match tensor rank ({inputTensor.Shape.Length})");
            }

            int[] start = new int[inputTensor.Shape.Length];
            int[] end = new int[inputTensor.Shape.Length];
            int[] step = new int[inputTensor.Shape.Length];
            bool[] isSlice = new bool[inputTensor.Shape.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                inputTensor.ParseIndex(indices[i], inputTensor.Shape[i], out start[i], out end[i], out step[i], out isSlice[i]);
            }

            Tensor result = new Tensor(inputTensor.Shape);
            Array.Fill(result.Data, 0.0f);  // Initialize with zeros

            int[] sourceIndices = new int[upstreamGradient.Shape.Length];
            int[] destIndices = new int[inputTensor.Shape.Length];

            this.CopyDataReverseOne(upstreamGradient, result, start, end, step, isSlice, sourceIndices, destIndices, 0);

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
        /// Copies data from the upstream gradient to the result tensor in reverse operation.
        /// </summary>
        /// <param name="source">The source tensor (upstream gradient).</param>
        /// <param name="dest">The destination tensor (original gradient).</param>
        /// <param name="start">The start indices for slicing.</param>
        /// <param name="end">The end indices for slicing.</param>
        /// <param name="step">The step sizes for slicing.</param>
        /// <param name="isSlice">Indicates whether the dimension is sliced.</param>
        /// <param name="sourceIndices">The current indices in the source tensor.</param>
        /// <param name="destIndices">The current indices in the destination tensor.</param>
        /// <param name="currentDim">The current dimension being processed.</param>
        private void CopyDataReverse(Tensor source, Tensor dest, int[] start, int[] end, int[] step, bool[] isSlice, Memory<int> sourceIndices, Memory<int> destIndices, int currentDim)
        {
            if (currentDim == dest.Shape.Length)
            {
                dest.Data[this.GetFullIndex(destIndices, 0, dest.Shape)] += source.Data[this.GetFullIndex(sourceIndices, 0, source.Shape)];
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
                    Memory<int> newSourceIndices = new int[sourceIndices.Length];
                    Memory<int> newDestIndices = new int[destIndices.Length];
                    sourceIndices.CopyTo(newSourceIndices);
                    destIndices.CopyTo(newDestIndices);
                    newSourceIndices.Span[currentDim] = destIndex;
                    newDestIndices.Span[currentDim] = i;

                    this.CopyDataReverse(source, dest, start, end, step, isSlice, newSourceIndices, newDestIndices, currentDim + 1);
                    destIndex++;
                }
            }
            else
            {
                Memory<int> newSourceIndices = new int[sourceIndices.Length];
                sourceIndices.CopyTo(newSourceIndices);
                newSourceIndices.Span[currentDim] = sourceStart;

                this.CopyDataReverse(source, dest, start, end, step, isSlice, newSourceIndices, destIndices, currentDim + 1);
            }
        }

        private void CopyDataReverseOne(Tensor source, Tensor dest, int[] start, int[] end, int[] step, bool[] isSlice, int[] sourceIndices, int[] destIndices, int currentDim)
        {
            int sourceSize = source.Shape[currentDim];
            int destStart = start[currentDim];
            int destEnd = end[currentDim];
            int destStep = step[currentDim];

            for (int i = 0; i < sourceSize; i++)
            {
                sourceIndices[currentDim] = i;
                destIndices[currentDim] = destStart + (i * destStep);

                if (destIndices[currentDim] < destEnd)
                {
                    if (currentDim < dest.Shape.Length - 1)
                    {
                        this.CopyDataReverseOne(source, dest, start, end, step, isSlice, sourceIndices, destIndices, currentDim + 1);
                    }
                    else if (destStep == 1)
                    {
                        // SIMD copy for the innermost dimension when step is 1
                        int vectorSize = PradTools.VectorCount();
                        int remainingSize = destEnd - destIndices[currentDim];
                        int vectorizableSize = remainingSize - (remainingSize % vectorSize);

                        int sourceIndex = this.GetFlatIndex(sourceIndices, source.Shape);
                        int destIndex = this.GetFlatIndex(destIndices, dest.Shape);

                        // Vector copy
                        for (int j = 0; j < vectorizableSize; j += vectorSize)
                        {
                            var sourceVector = PradTools.AllocateVector(source.Data, sourceIndex + j);
                            sourceVector.CopyTo(dest.Data, destIndex + j);
                        }

                        // Copy remaining elements
                        for (int j = vectorizableSize; j < remainingSize; j++)
                        {
                            dest.Data[destIndex + j] = source.Data[sourceIndex + j];
                        }

                        // Skip the rest of the innermost loop as we've copied everything
                        return;
                    }
                    else
                    {
                        // Regular copy for non-contiguous or non-SIMD cases
                        int sourceIndex = this.GetFlatIndex(sourceIndices, source.Shape);
                        int destIndex = this.GetFlatIndex(destIndices, dest.Shape);
                        dest.Data[destIndex] = source.Data[sourceIndex];
                    }
                }
            }
        }

        private int GetFlatIndex(int[] indices, int[] shape)
        {
            int index = 0;
            int stride = 1;
            for (int i = indices.Length - 1; i >= 0; i--)
            {
                index += indices[i] * stride;
                stride *= shape[i];
            }

            return index;
        }

        private int GetFullIndex2(int[] indices, int[] start, int[] shape)
        {
            int index = 0;
            int stride = 1;
            for (int i = indices.Length - 1; i >= 0; i--)
            {
                index += (indices[i] - start[i]) * stride;
                stride *= shape[i];
            }

            return index;
        }

        private int GetFullIndex(Memory<int> indices, int lastDimIndex, Memory<int> shape)
        {
            int index = 0;
            int stride = 1;
            for (int i = indices.Length - 1; i >= 0; i--)
            {
                if (i == indices.Length - 1)
                {
                    index += lastDimIndex * stride;
                }
                else
                {
                    index += indices.Span[i] * stride;
                }

                stride *= shape.Span[i];
            }

            return index;
        }

        private Tensor TileReverseSpecializedCase1NSIMD(Tensor upstreamGradient, int[] originalShape, int multipleSecondDim)
        {
            Tensor grad = new Tensor(originalShape);
            int firstDimSize = originalShape[0];
            int originalSecondDimSize = originalShape[1];
            int tiledSecondDimSize = originalSecondDimSize * multipleSecondDim;

            int vectorSize = Vector<double>.Count;
            int simdIterations = originalSecondDimSize / vectorSize;
            int remainderStart = simdIterations * vectorSize;

            Parallel.For(0, firstDimSize, i =>
            {
                int baseGradIndex = i * originalSecondDimSize;
                int baseUpstreamIndex = i * tiledSecondDimSize;

                for (int j = 0; j < simdIterations; j++)
                {
                    var sum = PradTools.VectorZero();
                    int gradIndex = baseGradIndex + (j * vectorSize);

                    for (int m = 0; m < multipleSecondDim; m++)
                    {
                        int upstreamIndex = baseUpstreamIndex + (m * originalSecondDimSize) + (j * vectorSize);
                        var vec = PradTools.AllocateVector(upstreamGradient.Data, upstreamIndex);
                        sum += vec;
                    }

                    sum.CopyTo(grad.Data, gradIndex);
                }

                // Handle remaining elements
                for (int j = remainderStart; j < originalSecondDimSize; j++)
                {
                    var sum = PradTools.Zero;
                    for (int m = 0; m < multipleSecondDim; m++)
                    {
                        int upstreamIndex = baseUpstreamIndex + (m * originalSecondDimSize) + j;
                        sum += upstreamGradient.Data[upstreamIndex];
                    }

                    grad.Data[baseGradIndex + j] = sum;
                }
            });

            return grad;
        }

        private Tensor TileReverseSpecializedCaseN1SIMD(Tensor upstreamGradient, int[] originalShape, int multipleFirstDim)
        {
            Tensor grad = new Tensor(originalShape);
            int originalFirstDimSize = originalShape[0];
            int secondDimSize = originalShape[1];
            int vectorSize = Vector<double>.Count;
            int simdIterations = secondDimSize / vectorSize;
            int remainderStart = simdIterations * vectorSize;

            Parallel.For(0, originalFirstDimSize, i =>
            {
                var sumArray = PradTools.AllocateArray(Math.Max(vectorSize, secondDimSize));
                int baseUpstreamIndex = i * multipleFirstDim * secondDimSize;
                int baseGradIndex = i * secondDimSize;

                for (int m = 0; m < multipleFirstDim; m++)
                {
                    int upstreamIndex = baseUpstreamIndex + (m * secondDimSize);

                    // SIMD summing
                    for (int j = 0; j < simdIterations; j++)
                    {
                        var vec = PradTools.AllocateVector(upstreamGradient.Data, upstreamIndex + (j * vectorSize));
                        var sumVec = PradTools.AllocateVector(sumArray);
                        sumVec += vec;
                        sumVec.CopyTo(sumArray);
                    }

                    // Handle remaining elements
                    for (int j = remainderStart; j < secondDimSize; j++)
                    {
                        sumArray[j] += upstreamGradient.Data[upstreamIndex + j];
                    }
                }

                // Copy only up to secondDimSize elements
                PradTools.AllocateSpan(sumArray, 0, secondDimSize)
                    .CopyTo(PradTools.AllocateSpan(grad.Data, baseGradIndex, secondDimSize));
            });

            return grad;
        }

        private Tensor TileReverseSpecializedCaseSecondDim(Tensor upstreamGradient, int[] originalShape, int multipleSecondDim)
        {
            Tensor grad = new Tensor(originalShape);
            int originalSecondDimSize = originalShape[1];
            int totalElementsPerSecondDim = originalShape[0] * originalShape[2] * originalShape[3];

            // Prepare SIMD vectors
            int vectorSize = Vector<double>.Count;
            int simdIterations = multipleSecondDim / vectorSize;
            int remainderElements = multipleSecondDim % vectorSize;

            Parallel.For(0, totalElementsPerSecondDim, i =>
            {
                int baseIndexOriginal = i * originalSecondDimSize;
                int baseIndexUpstream = i * multipleSecondDim;

                for (int j = 0; j < originalSecondDimSize; j++)
                {
                    var sum = PradTools.VectorZero();
                    int upstreamIndex = baseIndexUpstream + j;

                    // SIMD summing
                    for (int k = 0; k < simdIterations; k++)
                    {
                        var vec = PradTools.AllocateVector(upstreamGradient.Data, upstreamIndex + (k * vectorSize * originalSecondDimSize));
                        sum += vec;
                    }

                    // Sum the elements in the final vector
                    var result = Vector.Dot(sum, PradTools.VectorOne());

                    // Handle remaining elements
                    for (int k = simdIterations * vectorSize; k < multipleSecondDim; k++)
                    {
                        result += upstreamGradient.Data[upstreamIndex + (k * originalSecondDimSize)];
                    }

                    grad.Data[baseIndexOriginal + j] = result / multipleSecondDim;
                }
            });

            return grad;
        }

        private Tensor TileReverseSpecializedCaseSIMD(Tensor upstreamGradient, int[] originalShape, int multipleThirdDim)
        {
            Tensor grad = new Tensor(originalShape);
            int originalThirdDimSize = originalShape[2];
            int totalElementsPerThirdDim = originalShape[0] * originalShape[1] * originalShape[3];

            int vectorSize = Vector<double>.Count;
            int simdIterations = multipleThirdDim / vectorSize;
            int remainderElements = multipleThirdDim % vectorSize;

            Parallel.For(0, totalElementsPerThirdDim, i =>
            {
                int baseIndex = i * originalThirdDimSize;
                for (int j = 0; j < originalThirdDimSize; j++)
                {
                    var sum = PradTools.VectorZero();
                    int upstreamIndex = baseIndex + (j * multipleThirdDim);

                    // SIMD summing
                    for (int k = 0; k < simdIterations; k++)
                    {
                        var vec = PradTools.AllocateVector(upstreamGradient.Data, upstreamIndex + (k * vectorSize));
                        sum += vec;
                    }

                    // Sum the elements in the final vector
                    var scalarSum = Vector.Dot(sum, PradTools.VectorOne());

                    // Handle remaining elements
                    for (int k = simdIterations * vectorSize; k < multipleThirdDim; k++)
                    {
                        scalarSum += upstreamGradient.Data[upstreamIndex + k];
                    }

                    grad.Data[baseIndex + j] = scalarSum;
                }
            });

            return grad;
        }

        private int GetTotalSize(int[] shape)
        {
            return shape.Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Converts multi-dimensional indices to a flat index.
        /// </summary>
        /// <param name="indices">The multi-dimensional indices.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The flat index.</returns>
        private int GetIndex(int[] indices, int[] shape)
        {
            if (indices.Length != shape.Length)
            {
                throw new ArgumentException("Indices length does not match tensor shape.");
            }

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
        /// Converts a flat index to multi-dimensional indices.
        /// </summary>
        /// <param name="index">The flat index.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The multi-dimensional indices.</returns>
        private int[] GetMultiDimensionalIndices(int index, int[] shape)
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ReverseDiffVerticalSIMD(Tensor gradInput, Tensor gradOutput, int height, int width)
        {
            if (height <= 1)
            {
                return;
            }

            int vectorWidth = PradTools.VectorCount();

            for (int x = 0; x < width; x++)
            {
                int y = 0;
                for (; y <= height - vectorWidth - 1; y += vectorWidth)
                {
                    int baseIdx = (y * width) + x;
                    var gradVec = PradTools.AllocateVector(gradOutput.Data, baseIdx);

                    for (int i = 0; i < vectorWidth; i++)
                    {
                        int pos = ((y + i) * width) + x;
                        var grad = gradVec[i];

                        gradInput.Data[pos] -= grad;
                        gradInput.Data[pos + width] += grad;
                    }
                }

                // Scalar remainder
                for (; y < height - 1; y++)
                {
                    int idx = (y * width) + x;
                    var grad = gradOutput.Data[idx];
                    gradInput.Data[idx] -= grad;
                    gradInput.Data[idx + width] += grad;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ReverseDiffHorizontalSIMD(Tensor gradInput, Tensor gradOutput, int height, int width)
        {
            if (width <= 1)
            {
                return;
            }

            int vectorWidth = PradTools.VectorCount();

            for (int y = 0; y < height; y++)
            {
                int rowStart = y * width;
                int x = 0;
                for (; x <= width - vectorWidth - 2; x += vectorWidth)
                {
                    int baseIdx = rowStart + x;
                    var gradVec = PradTools.AllocateVector(gradOutput.Data, baseIdx);

                    for (int i = 0; i < vectorWidth; i++)
                    {
                        int pos = baseIdx + i;
                        var grad = gradVec[i];

                        gradInput.Data[pos] -= grad;
                        gradInput.Data[pos + 1] += grad;
                    }
                }

                // Scalar remainder
                for (; x < width - 1; x++)
                {
                    int idx = rowStart + x;
                    var grad = gradOutput.Data[idx];
                    gradInput.Data[idx] -= grad;
                    gradInput.Data[idx + 1] += grad;
                }
            }
        }

        /// <summary>
        /// Checks if the shapes of the tensors are compatible.
        /// </summary>
        /// <param name="tensorA">The first tensor.</param>
        /// <param name="tensorB">The second tensor.</param>
        private void CheckShapeCompatibility(Tensor tensorA, Tensor tensorB)
        {
            if (tensorA.Shape.Length != tensorB.Shape.Length)
            {
                throw new ArgumentException("Shapes are not compatible for the operation.");
            }

            for (int i = 0; i < tensorA.Shape.Length; i++)
            {
                if (tensorA.Shape[i] != tensorB.Shape[i])
                {
                    throw new ArgumentException("Shapes are not compatible for the operation.");
                }
            }
        }
    }
}

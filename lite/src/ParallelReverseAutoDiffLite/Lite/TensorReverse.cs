//------------------------------------------------------------------------------
// <copyright file="TensorReverse.cs" author="ameritusweb" date="6/20/2024">
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
    /// Backward functions for tensors.
    /// </summary>
    public partial class TensorReverse
    {
        /// <summary>
        /// Computes the reverse gradient of excluding.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The gradient of the exclusion operation.</returns>
        public Tensor ExcludeReverse(Tensor upstreamGradient, double min, double max)
        {
            Tensor tensor = this.InitialTensors[0];

            var gradData = new float[tensor.Data.Length];

            Parallel.For(0, tensor.Data.Length, i =>
            {
                if (tensor.Data[i] < min || tensor.Data[i] > max)
                {
                    gradData[i] = upstreamGradient.Data[i];
                }
                else
                {
                    gradData[i] = 0.0f;
                }
            });

            return new Tensor(tensor.Shape, gradData);
        }

        /// <summary>
        /// Computes the reverse gradient for clipping.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clipped tensor gradient.</returns>
        public Tensor ClipReverse(Tensor upstreamGradient, double min, double max)
        {
            Tensor tensor = this.InitialTensors[0];

            var gradData = new float[tensor.Data.Length];
            var minArray = new float[tensor.Data.Length];
            var maxArray = new float[tensor.Data.Length];
            var zeroArray = new float[tensor.Data.Length];

            // Fill minArray, maxArray and zeroArray with appropriate values
            Array.Fill(minArray, (float)min);
            Array.Fill(maxArray, (float)max);
            Array.Fill(zeroArray, 0.0f);

            // Create intermediate arrays for comparison
            var clippedMin = new float[tensor.Data.Length];
            var clippedMax = new float[tensor.Data.Length];
            var clippedMask = new float[tensor.Data.Length];

            // Perform element-wise min and max operations
            Vml.MinMag(tensor.Data.Length, tensor.Data, minArray, clippedMin);
            Vml.MaxMag(tensor.Data.Length, clippedMin, maxArray, clippedMax);

            // Determine the mask for valid positions (1 if valid, 0 if clipped)
            Parallel.For(0, tensor.Data.Length, i =>
            {
                clippedMask[i] = (clippedMax[i] == tensor.Data[i]) ? 1.0f : 0.0f;
            });

            // Apply the mask to the upstream gradient
            Vml.Mul(tensor.Data.Length, upstreamGradient.Data, clippedMask, gradData);

            return new Tensor(tensor.Shape, gradData);
        }

        /// <summary>
        /// Computes the reverse gradient for matrix multiplication.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradients with respect to the input tensors.</returns>
        public Tensor[] MatrixMultiplyReverse(Tensor upstreamGradient)
        {
            if (this.InitialTensors.Length != 2)
            {
                throw new InvalidOperationException("MatrixMultiplyReverse expects exactly two initial tensors.");
            }

            Tensor tensorA = this.InitialTensors[0];
            Tensor tensorB = this.InitialTensors[1];

            if (tensorA.Shape.Length != 2 || tensorB.Shape.Length != 2)
            {
                throw new ArgumentException("Matrix multiplication gradients are only supported for 2D tensors.");
            }

            int m = tensorA.Shape[0]; // Rows of A
            int k = tensorA.Shape[1]; // Columns of A and rows of B
            int n = tensorB.Shape[1]; // Columns of B

            if (k != tensorB.Shape[0])
            {
                throw new ArgumentException("Incompatible shapes for matrix multiplication.");
            }

            // Gradient w.r.t tensorA: dL/dA = dL/dC * B^T
            var gradAData = new float[m * k];
            var gradA = new Tensor(new int[] { m, k }, gradAData);

            MKLNET.Blas.gemm(
                Layout.RowMajor,
                Trans.No,
                Trans.Yes,
                m,
                k,
                n,
                1.0f,
                upstreamGradient.Data.AsSpan(),
                n,
                tensorB.Data.AsSpan(),
                n,
                0.0f,
                gradAData.AsSpan(),
                k);

            // Gradient w.r.t tensorB: dL/dB = A^T * dL/dC
            var gradBData = new float[k * n];
            var gradB = new Tensor(new int[] { k, n }, gradBData);

            MKLNET.Blas.gemm(
                Layout.RowMajor,
                Trans.Yes,
                Trans.No,
                k,
                n,
                m,
                1.0f,
                tensorA.Data.AsSpan(),
                k,
                upstreamGradient.Data.AsSpan(),
                n,
                0.0f,
                gradBData.AsSpan(),
                n);

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for concatenation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axis">The axis along which the concatenation was performed.</param>
        /// <returns>An array of tensors representing the gradients for each input tensor.</returns>
        public Tensor[] ConcatReverse(Tensor upstreamGradient, int axis)
        {
            int numTensors = this.InitialTensors.Length;
            Tensor[] gradients = new Tensor[numTensors];

            int[] shape = this.InitialTensors[0].Shape;
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

            // Split the upstream gradient along the concatenation axis
            int offset = 0;
            for (int i = 0; i < numTensors; i++)
            {
                int[] gradShape = (int[])shape.Clone();
                gradShape[axis] = this.InitialTensors[i].Shape[axis];

                int gradSize = gradShape.Aggregate((a, b) => a * b);
                float[] gradData = new float[gradSize];
                Array.Copy(upstreamGradient.Data, offset, gradData, 0, gradSize);

                gradients[i] = new Tensor(gradShape, gradData);
                offset += gradSize;
            }

            return gradients;
        }

        /// <summary>
        /// Computes the reverse gradient for the mean operation along the specified axis.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axis">The axis along which the mean was computed.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor MeanReverse(Tensor upstreamGradient, int axis)
        {
            Tensor inputTensor = this.InitialTensors[0];
            if (axis < 0)
            {
                axis += inputTensor.Shape.Length;
            }

            if (axis < 0 || axis >= inputTensor.Shape.Length)
            {
                throw new ArgumentException("Axis is out of bounds for the tensor.");
            }

            int[] outputShape = inputTensor.Shape.Where((_, idx) => idx != axis).ToArray();
            if (outputShape.Length == 0)
            {
                outputShape = new int[] { 1 };
            }

            int axisSize = inputTensor.Shape[axis];
            float scale = 1.0f / axisSize;
            int outerSize = inputTensor.Shape.Take(axis).Aggregate(1, (a, b) => a * b);
            int innerSize = inputTensor.Shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b);

            float[] gradData = new float[inputTensor.Data.Length];

            float[] scaleData = new float[innerSize];
            Array.Fill(scaleData, scale);

            Parallel.For(0, outerSize, outerIdx =>
            {
                int gradOffset = outerIdx * axisSize * innerSize;
                int upstreamOffset = outerIdx * innerSize;

                // Scale the upstream gradient
                float[] scaledUpstream = new float[innerSize];
                Vml.Mul(innerSize, upstreamGradient.Data.AsSpan(upstreamOffset, innerSize), scaleData, scaledUpstream);

                // Repeat the scaled gradient for each element along the axis
                for (int axisIdx = 0; axisIdx < axisSize; axisIdx++)
                {
                    Blas.copy(innerSize, scaledUpstream, 1, gradData.AsSpan(gradOffset + (axisIdx * innerSize)), 1);
                }
            });

            return new Tensor(inputTensor.Shape, gradData);
        }

        /// <summary>
        /// Reverses the split operation.
        /// </summary>
        /// <param name="upstreamGradients">The gradients from the split tensors.</param>
        /// <param name="axis">The axis along which the tensor was split.</param>
        /// <returns>The gradient with respect to the input tensor before splitting.</returns>
        /// <exception cref="ArgumentException">If the shapes of the upstream gradients are not compatible.</exception>
        public Tensor SplitReverse(Tensor[] upstreamGradients, int axis = 0)
        {
            if (upstreamGradients == null || upstreamGradients.Length == 0)
            {
                throw new ArgumentException("The input list of upstream gradients cannot be empty.");
            }

            int[] gradShape = upstreamGradients[0].Shape;
            int rank = gradShape.Length;
            int numTensors = upstreamGradients.Length;

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

            // Determine the shape of the resulting gradient tensor
            int[] resultShape = new int[rank];
            Array.Copy(gradShape, resultShape, rank);
            resultShape[axis] *= numTensors;

            // Calculate the total size of the result gradient tensor
            int totalSize = resultShape.Aggregate(1, (a, b) => a * b);
            float[] resultData = new float[totalSize];
            var result = new Tensor(resultShape, resultData);

            int[] strides = new int[rank];
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * resultShape[i + 1];
            }

            Parallel.For(0, numTensors, i =>
            {
                int start = i * strides[axis] * gradShape[axis];
                var gradient = upstreamGradients[i].Data;

                for (int j = 0; j < gradient.Length; j++)
                {
                    int[] indices = new int[rank];
                    int temp = j;
                    for (int k = rank - 1; k >= 0; k--)
                    {
                        indices[k] = temp % gradShape[k];
                        temp /= gradShape[k];
                    }

                    indices[axis] += i * gradShape[axis];

                    int resultIndex = 0;
                    for (int k = 0; k < rank; k++)
                    {
                        resultIndex += indices[k] * strides[k];
                    }

                    resultData[resultIndex] = gradient[j];
                }
            });

            return result;
        }

        /// <summary>
        /// The reverse unstack operation.
        /// </summary>
        /// <param name="upstreamGradients">The upstream gradients from the unstacked tensors.</param>
        /// <param name="axis">The axis along which the tensor was unstacked.</param>
        /// <returns>The gradient with respect to the input tensor before unstacking.</returns>
        /// <exception cref="ArgumentException">If the axis is out of bounds or upstream gradients are empty.</exception>
        public Tensor UnstackReverse(Tensor[] upstreamGradients, int axis = 0)
        {
            if (upstreamGradients == null || upstreamGradients.Length == 0)
            {
                throw new ArgumentException("The input list of upstream gradients cannot be empty.");
            }

            int[] gradShape = upstreamGradients[0].Shape;
            int numTensors = upstreamGradients.Length;
            int rank = gradShape.Length + 1; // We're adding back the unstacked dimension

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

            // Determine the shape of the resulting gradient tensor
            int[] resultShape = new int[rank];
            int gradIndex = 0;
            for (int i = 0; i < rank; i++)
            {
                if (i == axis)
                {
                    resultShape[i] = numTensors;
                }
                else
                {
                    resultShape[i] = gradShape[gradIndex++];
                }
            }

            // Calculate the total size of the result gradient tensor
            int totalSize = resultShape.Aggregate(1, (a, b) => a * b);
            float[] resultData = new float[totalSize];
            var result = new Tensor(resultShape, resultData);

            int[] strides = new int[rank];
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * resultShape[i + 1];
            }

            Parallel.For(0, numTensors, i =>
            {
                int start = i * strides[axis];
                var gradient = upstreamGradients[i].Data;

                for (int j = 0; j < gradient.Length; j++)
                {
                    int[] indices = new int[rank];
                    int temp = j;
                    int gradDim = 0;
                    for (int k = rank - 1; k >= 0; k--)
                    {
                        if (k != axis)
                        {
                            indices[k] = temp % gradShape[gradDim];
                            temp /= gradShape[gradDim];
                            gradDim++;
                        }
                    }

                    indices[axis] = i;

                    int resultIndex = 0;
                    for (int k = 0; k < rank; k++)
                    {
                        resultIndex += indices[k] * strides[k];
                    }

                    resultData[resultIndex] += gradient[j];
                }
            });

            return result;
        }

        /// <summary>
        /// Recursively expands the upstream gradient back to the original shape.
        /// </summary>
        /// <param name="inputIndex">The current indices in the input tensor.</param>
        /// <param name="axes">The axes along which the summation was performed.</param>
        /// <param name="currentAxis">The current axis being processed.</param>
        /// <param name="value">The value of the upstream gradient to be distributed.</param>
        /// <param name="strides">The strides of the input tensor.</param>
        /// <param name="result">The tensor to accumulate the gradient in.</param>
        private void SumReverseRecursive(int[] inputIndex, int[] axes, int currentAxis, float value, int[] strides, Tensor result)
        {
            if (currentAxis == inputIndex.Length)
            {
                int flatIndex = 0;
                for (int i = 0; i < inputIndex.Length; i++)
                {
                    flatIndex += inputIndex[i] * strides[i];
                }

                lock (result.Data)
                {
                    result.Data[flatIndex] += value;
                }
            }
            else
            {
                if (axes.Contains(currentAxis))
                {
                    for (int i = 0; i < result.Shape[currentAxis]; i++)
                    {
                        inputIndex[currentAxis] = i;
                        this.SumReverseRecursive(inputIndex, axes, currentAxis + 1, value, strides, result);
                    }
                }
                else
                {
                    this.SumReverseRecursive(inputIndex, axes, currentAxis + 1, value, strides, result);
                }
            }
        }
    }
}

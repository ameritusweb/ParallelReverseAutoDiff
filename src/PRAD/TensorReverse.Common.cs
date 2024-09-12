//------------------------------------------------------------------------------
// <copyright file="TensorReverse.Common.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
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

            // Gradient of abs(x) is 1 for x > 0, -1 for x < 0, and undefined for x = 0
            Tensor gradInput = new Tensor(inputTensor.Shape);
            Parallel.For(0, inputTensor.Data.Length, i =>
            {
                gradInput.Data[i] = inputTensor.Data[i] > 0 ? upstreamGradient.Data[i] :
                                    (inputTensor.Data[i] < 0 ? -upstreamGradient.Data[i] : 0);
            });

            return gradInput;
        }

        /// <summary>
        /// Computes the reverse gradient for broadcasting.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="originalShape">The original shape before broadcasting.</param>
        /// <returns>The gradient with respect to the input tensor before broadcasting.</returns>
        public Tensor BroadcastToReverse(Tensor upstreamGradient, int[] originalShape)
        {
            int[] broadcastShape = upstreamGradient.Shape;
            int[] newShape = new int[originalShape.Length];

            // Determine the new shape to sum the gradients along the broadcasted dimensions
            for (int i = 0; i < originalShape.Length; i++)
            {
                if (originalShape[i] == broadcastShape[broadcastShape.Length - originalShape.Length + i])
                {
                    newShape[i] = originalShape[i];
                }
                else
                {
                    newShape[i] = 1;
                }
            }

            Tensor result = upstreamGradient.Reshape(broadcastShape)
                .Sum(newShape);

            return result.Reshape(originalShape);
        }

        /// <summary>
        /// Computes the reverse gradient for summation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axes">The axes along which the summation was performed.</param>
        /// <returns>The gradient with respect to the input tensor before summation.</returns>
        public Tensor SumReverse(Tensor upstreamGradient, int[] axes)
        {
            var originalShape = this.InitialTensors[0].Shape;

            // Determine the shape after summation
            var summedShape = originalShape.ToList();
            foreach (var axis in axes.OrderByDescending(a => a))
            {
                summedShape.RemoveAt(axis);
            }

            if (summedShape.Count == 0)
            {
                summedShape.Add(1);
            }

            // Create a tensor with the shape after summation
            Tensor result = new Tensor(originalShape, PradTools.Zero);

            // Calculate strides for the original tensor
            var strides = new int[originalShape.Length];
            strides[originalShape.Length - 1] = 1;
            for (int i = originalShape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * originalShape[i + 1];
            }

            // Expand the upstream gradient back to the original shape
            Parallel.For(0, upstreamGradient.Data.Length, i =>
            {
                var upstreamIndex = new int[summedShape.Count];
                var inputIndex = new int[originalShape.Length];
                int remainingIndex = i;
                for (int j = summedShape.Count - 1; j >= 0; j--)
                {
                    upstreamIndex[j] = remainingIndex % summedShape[j];
                    remainingIndex /= summedShape[j];
                }

                for (int j = 0, k = 0; j < originalShape.Length; j++)
                {
                    if (axes.Contains(j))
                    {
                        inputIndex[j] = 0;
                    }
                    else
                    {
                        inputIndex[j] = upstreamIndex[k++];
                    }
                }

                this.SumReverseRecursive(inputIndex, axes, 0, upstreamGradient.Data[i], strides, result);
            });

            return result;
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

            if (padding == "SAME")
            {
                padTop = (filterHeight - 1) / 2;
                padBottom = (filterHeight - 1) - padTop;
                padLeft = (filterWidth - 1) / 2;
                padRight = (filterWidth - 1) - padLeft;
            }
            else if (padding == "VALID")
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
        /// Computes the reverse gradient for the ExpandDims operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ExpandDimsReverse(Tensor upstreamGradient, int axis = -1)
        {
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
            for (int i = 0; i < originalShape[0]; i++)
            {
                for (int j = 0; j < originalShape[1]; j++)
                {
                    grad[i, j] = upstreamGradient[i, 0];
                }
            }

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

            int[] newShape = inputTensor.CalculateNewShape(start, end, step, isSlice);
            Tensor result = new Tensor(inputTensor.Shape);

            this.CopyDataReverse(upstreamGradient, result, start, end, step, isSlice, new int[inputTensor.Shape.Length], new int[newShape.Length], 0);

            return result;
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
        private void CopyDataReverse(Tensor source, Tensor dest, int[] start, int[] end, int[] step, bool[] isSlice, int[] sourceIndices, int[] destIndices, int currentDim)
        {
            if (currentDim == dest.Shape.Length)
            {
                dest[destIndices] += source[sourceIndices];
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
                    newSourceIndices[currentDim] = destIndex;

                    int[] newDestIndices = (int[])destIndices.Clone();
                    newDestIndices[currentDim] = i;

                    this.CopyDataReverse(source, dest, start, end, step, isSlice, newSourceIndices, newDestIndices, currentDim + 1);
                    destIndex++;
                }
            }
            else
            {
                int[] newSourceIndices = (int[])sourceIndices.Clone();
                newSourceIndices[currentDim] = sourceStart;

                this.CopyDataReverse(source, dest, start, end, step, isSlice, newSourceIndices, destIndices, currentDim + 1);
            }
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

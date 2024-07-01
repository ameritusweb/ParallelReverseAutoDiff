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
    public class TensorReverse
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
            Tensor result = new Tensor(originalShape, 0.0f);

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
            Blas.scal(2.0f, gradA.Data);

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

            // Compute sqrt(x)
            Vml.Sqrt(x.Data.Length, x.Data, sqrtX.Data);

            // Compute gradX = upstreamGradient / (2 * sqrt(x))
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, sqrtX.Data, gradX.Data);
            Blas.scal(0.5f, gradX.Data);

            return gradX;
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
            Blas.scal(-1.0f, gradA.Data);

            return gradA;
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
            Vml.Pow(y.Data.Length, y.Data, Enumerable.Repeat(2.0f, y.Data.Length).ToArray(), ySquared.Data);
            Vml.Pow(x.Data.Length, x.Data, Enumerable.Repeat(2.0f, x.Data.Length).ToArray(), xSquared.Data);

            // Compute denominator = y^2 + x^2
            Vml.Add(ySquared.Data.Length, ySquared.Data, xSquared.Data, denominator.Data);

            // Compute gradY = upstreamGradient * x / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, x.Data, gradY.Data);
            Vml.Div(gradY.Data.Length, gradY.Data, denominator.Data, gradY.Data);

            // Compute gradX = -upstreamGradient * y / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, y.Data, gradX.Data);
            Vml.Div(gradX.Data.Length, gradX.Data, denominator.Data, gradX.Data);
            Blas.scal(-1.0f, gradX.Data);

            return new Tensor[] { gradY, gradX };
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

            Tensor negativeOnes = new Tensor(gradient.Shape, -1f);

            Vml.Mul(gradient.Data.Length, gradient.Data, negativeOnes.Data, gradient.Data);

            return gradient;
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
            Blas.scal(-1.0f, gradB.Data);

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
            Blas.scal(-1.0f, gradB.Data);

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

            strides = strides ?? new int[inputTensor.Shape.Length];
            for (int i = 0; i < inputTensor.Shape.Length; i++)
            {
                if (strides[i] == 0)
                {
                    strides[i] = 1;
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
                    sourceIndices[i] = begin[i] + (resultIndices[i] * strides[i]);
                }

                inputGradient[sourceIndices] += upstreamGradient[resultIndices];
            });

            return inputGradient;
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

            return result;
        }

        /// <summary>
        /// Computes the reverse gradient for the Indexer operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="indices">The indices used to slice.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor IndexerReverse(Tensor upstreamGradient, params string[] indices)
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

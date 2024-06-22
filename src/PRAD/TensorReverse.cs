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
        /// <param name="tensors">The transformed tensors.</param>
        public TensorReverse(Tensor[] tensors)
        {
            this.TransformedTensors = tensors;
        }

        /// <summary>
        /// Gets the transformed tensors.
        /// </summary>
        public Tensor[] TransformedTensors { get; private set; }

        /// <summary>
        /// Computes the reverse gradient for element-wise addition.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseAddReverse(Tensor upstreamGradient)
        {
            if (this.TransformedTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseAddReverse expects exactly two transformed tensors.");
            }

            Tensor tensorA = this.TransformedTensors[0];
            Tensor tensorB = this.TransformedTensors[1];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorB, upstreamGradient);

            // The gradient is the same for both input tensors
            Tensor gradA = upstreamGradient;
            Tensor gradB = upstreamGradient;

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise multiplication.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensors.</returns>
        public Tensor[] ElementwiseMultiplyReverse(Tensor upstreamGradient)
        {
            if (this.TransformedTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseMultiplyReverse expects exactly two transformed tensors.");
            }

            Tensor tensorA = this.TransformedTensors[0];
            Tensor tensorB = this.TransformedTensors[1];

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
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSquareReverse expects exactly one transformed tensor.");
            }

            Tensor tensorA = this.TransformedTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Vml.Mul(tensorA.Data.Length, upstreamGradient.Data, tensorA.Data, gradA.Data);
            Blas.scal(2.0, gradA.Data);

            return gradA;
        }

        /// <summary>
        /// Computes the reverse gradient for the element-wise square root operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseSquareRootReverse(Tensor upstreamGradient)
        {
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSquareRootReverse expects exactly one transformed tensor.");
            }

            Tensor y = this.TransformedTensors[0]; // The output of the square root operation

            this.CheckShapeCompatibility(y, upstreamGradient);

            var gradX = new Tensor(y.Shape);

            // Compute gradX = upstreamGradient / (2 * y)
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, y.Data, gradX.Data);
            Blas.scal(0.5, gradX.Data);

            return gradX;
        }

        /// <summary>
        /// Computes the reverse gradient for element-wise sine.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor ElementwiseSinReverse(Tensor upstreamGradient)
        {
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseSinReverse expects exactly one transformed tensor.");
            }

            Tensor tensorA = this.TransformedTensors[0];

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
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseCosReverse expects exactly one transformed tensor.");
            }

            Tensor tensorA = this.TransformedTensors[0];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);

            Tensor gradA = new Tensor(tensorA.Shape);
            Tensor sinTensor = new Tensor(tensorA.Shape);
            Vml.Sin(tensorA.Data.Length, tensorA.Data, sinTensor.Data);
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, sinTensor.Data, gradA.Data);
            Blas.scal(-1.0, gradA.Data);

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
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseAtan2Reverse expects exactly one transformed tensor.");
            }

            Tensor y = this.TransformedTensors[0];

            this.CheckShapeCompatibility(y, upstreamGradient);
            this.CheckShapeCompatibility(y, x);

            var gradY = new Tensor(y.Shape);
            var gradX = new Tensor(x.Shape);

            var ySquared = new Tensor(y.Shape);
            var xSquared = new Tensor(x.Shape);
            var denominator = new Tensor(y.Shape);

            // Compute y^2 and x^2
            Vml.Pow(y.Data.Length, y.Data, Enumerable.Repeat(2.0, y.Data.Length).ToArray(), ySquared.Data);
            Vml.Pow(x.Data.Length, x.Data, Enumerable.Repeat(2.0, x.Data.Length).ToArray(), xSquared.Data);

            // Compute denominator = y^2 + x^2
            Vml.Add(ySquared.Data.Length, ySquared.Data, xSquared.Data, denominator.Data);

            // Compute gradY = upstreamGradient * x / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, x.Data, gradY.Data);
            Vml.Div(gradY.Data.Length, gradY.Data, denominator.Data, gradY.Data);

            // Compute gradX = -upstreamGradient * y / denominator
            Vml.Mul(upstreamGradient.Data.Length, upstreamGradient.Data, y.Data, gradX.Data);
            Vml.Div(gradX.Data.Length, gradX.Data, denominator.Data, gradX.Data);
            Blas.scal(-1.0, gradX.Data);

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
            int numTensors = this.TransformedTensors.Length;
            Tensor[] gradients = new Tensor[numTensors];

            int[] shape = this.TransformedTensors[0].Shape;
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
                gradShape[axis] = this.TransformedTensors[i].Shape[axis];

                int gradSize = gradShape.Aggregate((a, b) => a * b);
                double[] gradData = new double[gradSize];
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
            int numTensors = this.TransformedTensors.Length;
            int[] shape = this.TransformedTensors[0].Shape;

            // Initialize the gradient tensors for each input tensor
            Tensor[] gradients = new Tensor[numTensors];
            for (int i = 0; i < numTensors; i++)
            {
                gradients[i] = new Tensor(shape);
            }

            int flatIndex = 0;

            for (int t = 0; t < numTensors; t++)
            {
                Tensor tensor = this.TransformedTensors[t];

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
            if (this.TransformedTensors == null || this.TransformedTensors.Length == 0)
            {
                throw new InvalidOperationException("The input list of transformed tensors cannot be empty.");
            }

            int numTensors = this.TransformedTensors.Length;
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
            int rank = gradShape.Length;

            // Handle negative axis
            if (axis < 0)
            {
                axis = rank + axis;
            }

            // Validate axis
            if (axis < 0 || axis > rank)
            {
                throw new ArgumentException("Axis value is out of bounds.");
            }

            // Determine the shape of the resulting gradient tensor
            int[] resultShape = new int[rank + 1];
            Array.Copy(gradShape, resultShape, rank);
            resultShape[axis] = numTensors;

            // Calculate the total size of the result gradient tensor
            int totalSize = resultShape.Aggregate(1, (a, b) => a * b);
            double[] resultData = new double[totalSize];
            var result = new Tensor(resultShape, resultData);

            int[] strides = new int[rank + 1];
            strides[rank] = 1;
            for (int i = rank - 1; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * resultShape[i + 1];
            }

            Parallel.For(0, numTensors, i =>
            {
                int start = i * strides[axis];
                var gradient = upstreamGradients[i].Data;

                for (int j = 0; j < gradient.Length; j++)
                {
                    int temp = j;
                    int resultIndex = start;
                    for (int k = rank - 1; k >= 0; k--)
                    {
                        resultIndex += (temp % gradShape[k]) * strides[k];
                        temp /= gradShape[k];
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
            Tensor inputTensor = this.TransformedTensors[0];
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
            if (this.TransformedTensors.Length != 2)
            {
                throw new InvalidOperationException("ElementwiseSubReverse expects exactly two transformed tensors.");
            }

            Tensor tensorA = this.TransformedTensors[0];
            Tensor tensorB = this.TransformedTensors[1];

            this.CheckShapeCompatibility(tensorA, upstreamGradient);
            this.CheckShapeCompatibility(tensorB, upstreamGradient);

            // Gradient with respect to A is just the upstream gradient
            Tensor gradA = upstreamGradient;

            // Gradient with respect to B is the negative of the upstream gradient
            Tensor gradB = new Tensor(upstreamGradient.Shape);
            Blas.copy(upstreamGradient.Data, gradB.Data);
            Blas.scal(-1.0, gradB.Data);

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
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("ElementwiseDivideReverse expects exactly one transformed tensor.");
            }

            Tensor a = this.TransformedTensors[0];
            Tensor b = other;

            this.CheckShapeCompatibility(a, upstreamGradient);
            this.CheckShapeCompatibility(a, b);

            var gradA = new Tensor(a.Shape);
            var gradB = new Tensor(b.Shape);

            // Compute gradA = upstreamGradient / b
            Vml.Div(upstreamGradient.Data.Length, upstreamGradient.Data, b.Data, gradA.Data);

            // Compute gradB = -upstreamGradient * a / (b * b)
            var aDivB = new Tensor(a.Shape);
            Vml.Div(a.Data.Length, a.Data, b.Data, aDivB.Data);

            Vml.Mul(aDivB.Data.Length, aDivB.Data, upstreamGradient.Data, gradB.Data);

            // Element-wise multiplication of upstreamGradient with -aDivB
            Blas.scal(-1.0, gradB.Data);

            var bSquared = new Tensor(b.Shape);
            Vml.Mul(b.Data.Length, b.Data, b.Data, bSquared.Data);
            Vml.Div(gradB.Data.Length, gradB.Data, bSquared.Data, gradB.Data);

            return new Tensor[] { gradA, gradB };
        }

        /// <summary>
        /// Computes the reverse gradient for the Tile operation.
        /// </summary>
        /// <param name="upstreamGradient">The gradient flowing from the upstream layer.</param>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>The gradient with respect to the input tensor.</returns>
        public Tensor TileReverse(Tensor upstreamGradient, int[] multiples)
        {
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("TileReverse expects exactly one transformed tensor.");
            }

            Tensor originalTensor = this.TransformedTensors[0];
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
            int[] originalIndices = new int[originalShape.Length];
            int[] tiledIndices = new int[tiledShape.Length];

            do
            {
                // Calculate corresponding indices in the original tensor
                for (int i = 0; i < originalShape.Length; i++)
                {
                    originalIndices[i] = tiledIndices[i] % originalShape[i];
                }

                // Accumulate gradient
                grad[originalIndices] += upstreamGradient[tiledIndices];

                // Move to the next index in the tiled tensor
                for (int i = tiledShape.Length - 1; i >= 0; i--)
                {
                    if (++tiledIndices[i] < tiledShape[i])
                    {
                        break;
                    }

                    tiledIndices[i] = 0;
                }
            }
            while (!tiledIndices.All(x => x == 0));

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
            Tensor inputTensor = this.TransformedTensors[0];

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
            Tensor inputTensor = this.TransformedTensors[0];

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
            int rank = gradShape.Length + 1;
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
            Array.Copy(gradShape, resultShape, axis);
            resultShape[axis] = gradShape[axis - 1] * numTensors;
            Array.Copy(gradShape, axis, resultShape, axis + 1, gradShape.Length - axis);

            // Calculate the total size of the result gradient tensor
            int totalSize = resultShape.Aggregate(1, (a, b) => a * b);
            double[] resultData = new double[totalSize];
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
                    int temp = j;
                    int resultIndex = start;
                    for (int k = gradShape.Length - 1; k >= 0; k--)
                    {
                        resultIndex += (temp % gradShape[k]) * strides[k + 1];
                        temp /= gradShape[k];
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
            Tensor transformedTensor = this.TransformedTensors[0];
            int[] originalShape = transformedTensor.Shape;

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
            if (this.TransformedTensors.Length != 1)
            {
                throw new InvalidOperationException("TransposeReverse expects exactly one transformed tensor.");
            }

            Tensor originalTensor = this.TransformedTensors[0];

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

            // Calculate the shape of the result tensor
            var newShape = permutation.Select(p => originalTensor.Shape[p]).ToArray();
            var result = new Tensor(newShape);

            // Perform the reverse transposition
            Parallel.For(0, result.Data.Length, i =>
            {
                int[] newIndices = result.GetMultiDimensionalIndices(i, newShape);
                int[] oldIndices = new int[newIndices.Length];
                for (int j = 0; j < newIndices.Length; j++)
                {
                    oldIndices[permutation[j]] = newIndices[j];
                }

                int oldIndex = originalTensor.GetIndex(oldIndices);
                result.Data[i] = upstreamGradient.Data[oldIndex];
            });

            return result;
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

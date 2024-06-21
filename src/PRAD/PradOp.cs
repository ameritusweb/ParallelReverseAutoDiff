//------------------------------------------------------------------------------
// <copyright file="PradOp.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// A lightweight reverse-mode automatic differentiation library.
    /// </summary>
    public class PradOp
    {
        private readonly List<(Func<Tensor, Tensor> backpropStep, PradResult result)> backpropagationSteps;
        private readonly Tensor seed;
        private Tensor currentTensor;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        /// <param name="seed">The seed tensor.</param>
        public PradOp(Tensor seed)
        {
            this.seed = seed;
            this.currentTensor = seed;
            this.backpropagationSteps = new List<(Func<Tensor, Tensor>, PradResult)>();
        }

        /// <summary>
        /// Adds two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to add.</param>
        /// <returns>The result of the addition along with the gradient placeholder.</returns>
        public PradResult Add(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseAdd(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseAddReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, grad.Data, 0, gradients[0].Data.Length * sizeof(double));
                return gradients[0];
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Subtracts a tensor from the current tensor element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to subtract.</param>
        /// <returns>The result of the subtraction along with the gradient placeholder.</returns>
        public PradResult Sub(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseSub(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseSubReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, grad.Data, 0, gradients[0].Data.Length * sizeof(double));
                return gradients[0];
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Multiplies two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to multiply with.</param>
        /// <returns>The result of the multiplication along with the gradient placeholder.</returns>
        public PradResult Mul(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseMultiply(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var gradA = new Tensor(this.currentTensor.Shape);
            var gradB = new Tensor(tensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseMultiplyReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, gradA.Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, gradB.Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients[0];
            };

            var pradResult = new PradResult(result, gradA);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the sine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the sine operation along with the gradient placeholder.</returns>
        public PradResult Sin()
        {
            var result = this.currentTensor.ElementwiseSin();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSinReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the cosine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the cosine operation along with the gradient placeholder.</returns>
        public PradResult Cos()
        {
            var result = this.currentTensor.ElementwiseCos();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseCosReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Reshapes the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="newShape">The new shape of the tensor.</param>
        /// <returns>The reshaped tensor along with the gradient placeholder.</returns>
        public PradResult Reshape(int[] newShape)
        {
            var result = this.currentTensor.Reshape(newShape);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ReshapeReverse(upstreamGrad, this.currentTensor.Shape);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Tiles the tensor along each dimension and records the operation for backpropagation.
        /// </summary>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>The tiled tensor along with the gradient placeholder.</returns>
        public PradResult Tile(int[] multiples)
        {
            var result = this.currentTensor.Tile(multiples);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.TileReverse(upstreamGrad, multiples);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Gathers slices from the tensor along the specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="indices">The indices of elements to gather.</param>
        /// <param name="axis">The axis along which to gather slices.</param>
        /// <returns>The gathered tensor along with the gradient placeholder.</returns>
        public PradResult Gather(Tensor indices, int axis = 0)
        {
            var result = this.currentTensor.Gather(indices, axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.GatherReverse(upstreamGrad, indices, axis);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Extracts a slice from the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="begin">The starting indices for each axis.</param>
        /// <param name="size">The lengths of the slice along each axis.</param>
        /// <param name="strides">The step size for each axis (default is 1).</param>
        /// <returns>The sliced tensor along with the gradient placeholder.</returns>
        public PradResult Slice(int[] begin, int[] size, int[]? strides = null)
        {
            var result = this.currentTensor.Slice(begin, size, strides);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = new Tensor(this.currentTensor.Shape);
            Func<Tensor, Tensor> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SliceReverse(upstreamGrad, begin, size, strides);
                Buffer.BlockCopy(gradient.Data, 0, grad.Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the backpropagation to accumulate gradients.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient flowing from the loss function.</param>
        public void Back(Tensor upstreamGradient)
        {
            Tensor currentUpstream = upstreamGradient;

            // Reverse iterate over backpropagation steps to accumulate gradients
            foreach (var (step, result) in this.backpropagationSteps.AsEnumerable().Reverse())
            {
                currentUpstream = step(currentUpstream);
                Buffer.BlockCopy(currentUpstream.Data, 0, result.Gradient.Data, 0, currentUpstream.Data.Length * sizeof(double));
            }
        }
    }
}

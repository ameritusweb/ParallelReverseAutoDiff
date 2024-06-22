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
        private readonly List<(Func<Tensor, Tensor[]> backpropStep, PradResult result)> backpropagationSteps;
        private readonly Tensor seed;
        private (Func<Tensor[], Tensor> splitStep, PradSplitResult result)? splitStep;
        private Tensor currentTensor;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        /// <param name="seed">The seed tensor.</param>
        public PradOp(Tensor seed)
        {
            this.seed = seed;
            this.currentTensor = seed;
            this.backpropagationSteps = new List<(Func<Tensor, Tensor[]>, PradResult)>();
        }

        /// <summary>
        /// Print code for the current tensor.
        /// </summary>
        /// <returns>The C# code.</returns>
        public string PrintCodeForCurrentTensor()
        {
            return this.currentTensor.PrintCode();
        }

        /// <summary>
        /// Creates a flat array from the tensors along the specified indices and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>The flat array along with the gradient placeholders.</returns>
        public PradResult CreateFlatArray(Tensor[] tensors, int[] indices)
        {
            var allTensors = tensors.Prepend(this.currentTensor).ToArray();
            var result = Tensor.CreateFlatArray(allTensors, indices);
            var tensorReverse = new TensorReverse(allTensors);

            var grad = Tensor.ToTensorArray(allTensors.Length, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.CreateFlatArrayReverse(upstreamGrad, indices);
                for (int i = 0; i < allTensors.Length; ++i)
                {
                    Buffer.BlockCopy(gradient[i].Data, 0, grad[i].Data, 0, gradient[i].Data.Length * sizeof(double));
                }

                return gradient;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
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

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseAddReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, grad[0].Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, grad[1].Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients;
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
        /// <returns>The result of the subtraction along with the gradient placeholders.</returns>
        public PradResult Sub(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseSub(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseSubReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, grad[0].Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, grad[1].Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients;
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
        /// <returns>The result of the multiplication along with the gradient placeholders.</returns>
        public PradResult Mul(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseMultiply(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseMultiplyReverse(upstreamGrad);
                Buffer.BlockCopy(gradients[0].Data, 0, grad[0].Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, grad[1].Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Divides two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to divide with.</param>
        /// <returns>The result of the division along with the gradient placeholders.</returns>
        public PradResult Div(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseDivide(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseDivideReverse(upstreamGrad, tensor);
                Buffer.BlockCopy(gradients[0].Data, 0, grad[0].Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, grad[1].Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the sine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the sine operation along with the gradient placeholders.</returns>
        public PradResult Sin()
        {
            var result = this.currentTensor.ElementwiseSin();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSinReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the cosine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the cosine operation along with the gradient placeholders.</returns>
        public PradResult Cos()
        {
            var result = this.currentTensor.ElementwiseCos();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseCosReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
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
        /// <returns>The reshaped tensor along with the gradient placeholders.</returns>
        public PradResult Reshape(int[] newShape)
        {
            var result = this.currentTensor.Reshape(newShape);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            var shape = newShape;
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ReshapeReverse(upstreamGrad, shape);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Splits the tensor into multiple tensors along the specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="groupSize">The group size.</param>
        /// <param name="axis">The axis along which to split.</param>
        /// <returns>The tensors along with the gradient placeholders.</returns>
        public PradOp[] Split(int groupSize, int axis = 0)
        {
            var results = this.currentTensor.Split(groupSize, axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor[], Tensor> splitStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SplitReverse(upstreamGrad, axis);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return gradient;
            };

            var ops = results.Select(x => new PradOp(x)).ToArray();
            this.splitStep = (splitStep, new PradSplitResult(results, grad));
            return ops;
        }

        /// <summary>
        /// Tiles the tensor along each dimension and records the operation for backpropagation.
        /// </summary>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>The tiled tensor along with the gradient placeholders.</returns>
        public PradResult Tile(int[] multiples)
        {
            var result = this.currentTensor.Tile(multiples);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.TileReverse(upstreamGrad, multiples);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
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
        /// <returns>The gathered tensor along with the gradient placeholders.</returns>
        public PradResult Gather(Tensor indices, int axis = 0)
        {
            var result = this.currentTensor.Gather(indices, axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.GatherReverse(upstreamGrad, indices, axis);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Gathers slices from the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="indices">The indices of elements to gather.</param>
        /// <returns>The gathered tensor along with the gradient placeholders.</returns>
        public PradResult GatherNd(Tensor indices)
        {
            var result = this.currentTensor.GatherNd(indices);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.GatherNdReverse(upstreamGrad, indices);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
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
        /// <returns>The sliced tensor along with the gradient placeholders.</returns>
        public PradResult Slice(int[] begin, int[] size, int[]? strides = null)
        {
            var result = this.currentTensor.Slice(begin, size, strides);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SliceReverse(upstreamGrad, begin, size, strides);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the arctangent of the quotient of the tensors' corresponding elements.
        /// </summary>
        /// <param name="tensor">The tensor to use as the divisor.</param>
        /// <returns>The result of the atan2 operation along with the gradient placeholders.</returns>
        public PradResult Atan2(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseAtan2(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseAtan2Reverse(upstreamGrad, tensor);
                Buffer.BlockCopy(gradients[0].Data, 0, grad[0].Data, 0, gradients[0].Data.Length * sizeof(double));
                Buffer.BlockCopy(gradients[1].Data, 0, grad[1].Data, 0, gradients[1].Data.Length * sizeof(double));
                return gradients;
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise square of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the square operation along with the gradient placeholders.</returns>
        public PradResult Square()
        {
            var result = this.currentTensor.ElementwiseSquare();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSquareReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise square root of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the square root operation along with the gradient placeholders.</returns>
        public PradResult SquareRoot()
        {
            var result = this.currentTensor.ElementwiseSquareRoot();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSquareRootReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Sums the rows of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The tensor with summed rows along with the gradient placeholders.</returns>
        public PradResult SumRows()
        {
            var result = this.currentTensor.SumRows();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SumRowsReverse(upstreamGrad);
                Buffer.BlockCopy(gradient.Data, 0, grad[0].Data, 0, gradient.Data.Length * sizeof(double));
                return new Tensor[] { gradient };
            };

            var pradResult = new PradResult(result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Stacks the current tensor with other tensors along a new axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensors">The tensors to stack.</param>
        /// <param name="axis">The axis along which to stack.</param>
        /// <returns>The stacked tensor along with the gradient placeholders.</returns>
        public PradResult Stack(Tensor[] tensors, int axis = 0)
        {
            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);

            var result = Tensor.Stack(combinedTensors, axis);
            var tensorReverse = new TensorReverse(combinedTensors);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.StackReverse(upstreamGrad, axis);
                for (int i = 0; i < gradients.Length; i++)
                {
                    Buffer.BlockCopy(gradients[i].Data, 0, grads[i].Data, 0, gradients[i].Data.Length * sizeof(double));
                }

                return gradients;
            };

            var pradResult = new PradResult(result, grads);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Concatenates the current tensor with other tensors along a specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensors">The tensors to concatenate.</param>
        /// <param name="axis">The axis along which to concatenate.</param>
        /// <returns>The concatenated tensor along with the gradient placeholders.</returns>
        public PradResult Concat(Tensor[] tensors, int axis = 0)
        {
            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);

            var result = Tensor.Concat(combinedTensors, axis);
            var tensorReverse = new TensorReverse(combinedTensors);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, Tensor[]> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ConcatReverse(upstreamGrad, axis);
                for (int i = 0; i < gradients.Length; i++)
                {
                    Buffer.BlockCopy(gradients[i].Data, 0, grads[i].Data, 0, gradients[i].Data.Length * sizeof(double));
                }

                return gradients;
            };

            var pradResult = new PradResult(result, grads);
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
                var gradients = step(currentUpstream);
                currentUpstream = gradients[0];
                int i = 0;
                foreach (var grad in result.Gradients)
                {
                    Buffer.BlockCopy(gradients[i].Data, 0, grad.Data, 0, gradients[i].Data.Length * sizeof(double));
                    i++;
                }
            }
        }

        /// <summary>
        /// Computes the backpropagation to accumulate gradients.
        /// </summary>
        /// <param name="upstreamGradients">The upstream gradients flowing from the loss function.</param>
        public void Back(Tensor[] upstreamGradients)
        {
            Tensor currentUpstream = this.splitStep!.Value.splitStep(upstreamGradients);

            // Reverse iterate over backpropagation steps to accumulate gradients
            foreach (var (step, result) in this.backpropagationSteps.AsEnumerable().Reverse())
            {
                var gradients = step(currentUpstream);
                currentUpstream = gradients[0];
                int i = 0;
                foreach (var grad in result.Gradients)
                {
                    Buffer.BlockCopy(gradients[i].Data, 0, grad.Data, 0, gradients[i].Data.Length * sizeof(double));
                    i++;
                }
            }
        }
    }
}

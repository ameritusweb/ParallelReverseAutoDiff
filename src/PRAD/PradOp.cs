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
    using System.Reflection;
    using System.Threading.Tasks;

    /// <summary>
    /// A lightweight reverse-mode automatic differentiation library.
    /// </summary>
    public class PradOp
    {
        private static PradOp funcOp = new PradOp();
        private readonly Tensor seed;
        private readonly Dictionary<Delegate, Delegate> operations;
        private List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)> backpropagationSteps;
        private (Func<Tensor[], Tensor> splitStep, PradSplitResult result)? splitStep;
        private Tensor currentTensor;
        private PradResult parentResult;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        /// <param name="seed">The seed tensor.</param>
        public PradOp(Tensor seed)
        {
            this.seed = seed;
            this.currentTensor = seed;
            this.backpropagationSteps = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();
            this.operations = new Dictionary<Delegate, Delegate>();
            this.InitializeOperations();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        internal PradOp()
        {
        }

        /// <summary>
        /// Gets the square root op.
        /// </summary>
        public static Func<PradResult> SquareRootOp => FuncOp.SquareRoot;

        /// <summary>
        /// Gets the add op.
        /// </summary>
        public static Func<Tensor, PradResult> AddOp => FuncOp.Add;

        /// <summary>
        /// Gets the mul op.
        /// </summary>
        public static Func<Tensor, PradResult> MulOp => FuncOp.Mul;

        /// <summary>
        /// Gets the sub op.
        /// </summary>
        public static Func<Tensor, PradResult> SubOp => FuncOp.Sub;

        /// <summary>
        /// Gets the div op.
        /// </summary>
        public static Func<Tensor, PradResult> DivOp => FuncOp.Div;

        /// <summary>
        /// Gets the sin op.
        /// </summary>
        public static Func<PradResult> SinOp => FuncOp.Sin;

        /// <summary>
        /// Gets the cos op.
        /// </summary>
        public static Func<PradResult> CosOp => FuncOp.Cos;

        /// <summary>
        /// Gets the GatherNd op.
        /// </summary>
        public static Func<Tensor, PradResult> GatherNdOp => FuncOp.GatherNd;

        /// <summary>
        /// Gets the add op.
        /// </summary>
        public static Func<PradResult> SumRowsOp => FuncOp.SumRows;

        /// <summary>
        /// Gets the square op.
        /// </summary>
        public static Func<PradResult> SquareOp => FuncOp.Square;

        /// <summary>
        /// Gets the atan2 op.
        /// </summary>
        public static Func<Tensor, PradResult> Atan2Op => FuncOp.Atan2;

        /// <summary>
        /// Gets the stack op.
        /// </summary>
        public static Func<Tensor[], int, PradResult> StackOp => FuncOp.Stack;

        /// <summary>
        /// Gets the concat op.
        /// </summary>
        public static Func<Tensor[], int, PradResult> ConcatOp => FuncOp.Concat;

        /// <summary>
        /// Gets the upstream gradient.
        /// </summary>
        public Tensor UpstreamGradient { get; private set; }

        /// <summary>
        /// Gets the seed gradient.
        /// </summary>
        public Tensor SeedGradient { get; internal set; }

        /// <summary>
        /// Gets a value indicating whether this is a dependent branch. If so, Back should not be called.
        /// </summary>
        public bool IsDependentBranch => this.parentResult != null;

        /// <summary>
        /// Gets an operation to get a func.
        /// </summary>
        internal static PradOp FuncOp => funcOp;

        /// <summary>
        /// Sets the upstream gradient.
        /// </summary>
        /// <param name="gradient">The upstream gradient.</param>
        /// <exception cref="ArgumentException">Shapes not matching.</exception>
        public void SetUpstreamGradient(Tensor gradient)
        {
            if (!gradient.Shape.SequenceEqual(this.currentTensor.Shape))
            {
                throw new ArgumentException("Upstream gradient shape must match the current tensor shape.");
            }

            this.UpstreamGradient = gradient;
        }

        /// <summary>
        /// Branch to another prad op.
        /// </summary>
        /// <returns>The other prad op.</returns>
        public PradOp Branch()
        {
            var branchedOp = new PradOp(this.currentTensor);
            branchedOp.parentResult = this.backpropagationSteps.Last().result;
            branchedOp.parentResult.Branches.Add(branchedOp);
            return branchedOp;
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
        /// Creates a deep clone of the current PradOp object.
        /// </summary>
        /// <returns>A new PradOp object that is a deep copy of the current instance.</returns>
        public PradOp DeepClone()
        {
            // Create a new PradOp instance with a deep clone of the seed tensor
            var clonedOp = new PradOp(this.seed.DeepClone());

            // Deep clone the current tensor
            clonedOp.currentTensor = this.currentTensor.DeepClone();

            // Deep clone the backpropagation steps
            clonedOp.backpropagationSteps = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();
            foreach (var (backpropStep, result) in this.backpropagationSteps)
            {
                var clonedResult = new PradResult(
                    clonedOp,
                    result.Result.DeepClone(),
                    result.Gradients.Select(g => g.DeepClone()).ToArray());
                clonedOp.backpropagationSteps.Add((backpropStep, clonedResult));
            }

            // Deep clone the split step if it exists
            if (this.splitStep.HasValue)
            {
                var (splitStep, splitResult) = this.splitStep.Value;
                var clonedSplitResult = new PradSplitResult(
                    splitResult.Results.Select(r => r.DeepClone()).ToArray(),
                    splitResult.Gradients.Select(g => g.DeepClone()).ToArray());
                clonedOp.splitStep = (splitStep, clonedSplitResult);
            }

            return clonedOp;
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
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.CreateFlatArrayReverse(upstreamGrad, indices);
                PradOp?[] ops = new PradOp?[allTensors.Length];
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = allTensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Applies a custom operation to the current tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="operation">The function that performs the operation.</param>
        /// <param name="reverseOperation">The function that computes the gradient of the operation.</param>
        /// <param name="outputShape">The shape of the output tensor.</param>
        /// <returns>The result of the custom operation along with the gradient placeholders.</returns>
        public PradResult CustomOperation(
            Func<Tensor, Tensor> operation,
            Func<Tensor, Tensor, Tensor, Tensor[]> reverseOperation,
            int[] outputShape)
        {
            var input = this.currentTensor.DeepClone();

            // Apply the operation to the current tensor
            var result = operation(this.currentTensor);

            // Initialize gradient tensors
            var grad = Tensor.ToTensorArray(1, outputShape);

            // Define the backpropagation step function
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = reverseOperation(input, result, upstreamGrad);
                PradOp?[] ops = new PradOp?[1] { null };
                return (gradients, ops);
            };

            // Create the PradResult object with the result and gradient placeholders
            var pradResult = new PradResult(this, result, grad);

            // Record the backpropagation step
            this.backpropagationSteps.Add((backpropStep, pradResult));

            // Update the current tensor
            this.currentTensor = result;

            return pradResult;
        }

        /// <summary>
        /// Adds two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to add.</param>
        /// <returns>The result of the addition along with the gradient placeholder.</returns>
        [PradOperation(nameof(AddOp))]
        public PradResult Add(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseAdd(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseAddReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Subtracts a tensor from the current tensor element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to subtract.</param>
        /// <returns>The result of the subtraction along with the gradient placeholders.</returns>
        [PradOperation(nameof(SubOp))]
        public PradResult Sub(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseSub(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseSubReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Multiplies two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to multiply with.</param>
        /// <returns>The result of the multiplication along with the gradient placeholders.</returns>
        [PradOperation(nameof(MulOp))]
        public PradResult Mul(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseMultiply(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseMultiplyReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Divides two tensors element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to divide with.</param>
        /// <returns>The result of the division along with the gradient placeholders.</returns>
        [PradOperation(nameof(DivOp))]
        public PradResult Div(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseDivide(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseDivideReverse(upstreamGrad, tensor);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the sine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the sine operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(SinOp))]
        public PradResult Sin()
        {
            var result = this.currentTensor.ElementwiseSin();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSinReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the cosine of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the cosine operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(CosOp))]
        public PradResult Cos()
        {
            var result = this.currentTensor.ElementwiseCos();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseCosReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Slices the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="indices">The indices of the elements to slice.</param>
        /// <returns>The result of the slice operation along with the gradient placeholders.</returns>
        public PradResult Indexer(params string[] indices)
        {
            var result = this.currentTensor.Indexer(indices);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.IndexerReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
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
            var oldShape = this.currentTensor.Shape;
            var result = this.currentTensor.Reshape(newShape);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ReshapeReverse(upstreamGrad, oldShape);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Transposes the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="permutations">The permutations.</param>
        /// <returns>The transposed tensor along with the gradient placeholders.</returns>
        public PradResult Transpose(params int[] permutations)
        {
            var result = this.currentTensor.Transpose(permutations);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.TransposeReverse(upstreamGrad, permutations);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
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
                return tensorReverse.SplitReverse(upstreamGrad, axis);
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
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.TileReverse(upstreamGrad, multiples);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
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
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.GatherReverse(upstreamGrad, indices, axis);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Gathers slices from the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="indices">The indices of elements to gather.</param>
        /// <returns>The gathered tensor along with the gradient placeholders.</returns>
        [PradOperation(nameof(GatherNdOp))]
        public PradResult GatherNd(Tensor indices)
        {
            var result = this.currentTensor.GatherNd(indices);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.GatherNdReverse(upstreamGrad, indices);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
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
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SliceReverse(upstreamGrad, begin, size, strides);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the arctangent of the quotient of the tensors' corresponding elements.
        /// </summary>
        /// <param name="tensor">The tensor to use as the divisor.</param>
        /// <returns>The result of the atan2 operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(Atan2Op))]
        public PradResult Atan2(Tensor tensor)
        {
            var result = this.currentTensor.ElementwiseAtan2(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseAtan2Reverse(upstreamGrad, tensor);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise square of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the square operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(SquareOp))]
        public PradResult Square()
        {
            var result = this.currentTensor.ElementwiseSquare();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSquareReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise square root of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the square root operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(SquareRootOp))]
        public PradResult SquareRoot()
        {
            var result = this.currentTensor.ElementwiseSquareRoot();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseSquareRootReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Sums the rows of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The tensor with summed rows along with the gradient placeholders.</returns>
        [PradOperation(nameof(SumRowsOp))]
        public PradResult SumRows()
        {
            var result = this.currentTensor.SumRows();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SumRowsReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
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
        [PradOperation(nameof(StackOp))]
        public PradResult Stack(Tensor[] tensors, int axis = 0)
        {
            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);

            var result = Tensor.Stack(combinedTensors, axis);
            var tensorReverse = new TensorReverse(combinedTensors);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.StackReverse(upstreamGrad, axis);
                PradOp?[] ops = new PradOp?[combinedTensors.Length];
                for (int i = 0; i < grads.Length; i++)
                {
                    var tensor = combinedTensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grads);
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
        [PradOperation(nameof(ConcatOp))]
        public PradResult Concat(Tensor[] tensors, int axis = 0)
        {
            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);

            var result = Tensor.Concat(combinedTensors, axis);
            var tensorReverse = new TensorReverse(combinedTensors);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ConcatReverse(upstreamGrad, axis);
                PradOp?[] ops = new PradOp?[combinedTensors.Length];
                for (int i = 0; i < grads.Length; i++)
                {
                    var tensor = combinedTensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grads);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Executes multiple operations in parallel and returns the results.
        /// </summary>
        /// <param name="operations">The operations to perform in parallel.</param>
        /// <returns>The results.</returns>
        /// <exception cref="ArgumentException">No operation provided.</exception>
        public PradResult[] DoParallel(params Func<PradOp, PradResult>[] operations)
        {
            if (operations == null || operations.Length == 0)
            {
                throw new ArgumentException("At least one operation must be provided.", nameof(operations));
            }

            var pradOps = new PradOp[operations.Length];
            pradOps[0] = this; // Use the calling PradOp for the first operation

            // Create branched PradOps for subsequent operations
            for (int i = 1; i < operations.Length; i++)
            {
                pradOps[i] = this.Branch();
            }

            var results = new PradResult[operations.Length];

            // Use Parallel.For to execute operations in parallel
            Parallel.For(0, operations.Length, i =>
            {
                results[i] = operations[i](pradOps[i]);
            });

            return results;
        }

        /// <summary>
        /// Computes backpropagation.
        /// </summary>
        /// <param name="tensor">The upstream gradient.</param>
        /// <returns>The gradient.</returns>
        public Tensor Back(Tensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException("The upstream gradient must be provided.");
            }

            if (this.IsDependentBranch)
            {
                throw new InvalidOperationException("Back should not be called on a dependent branch.");
            }

            this.UpstreamGradient = tensor;
            return this.Back();
        }

        /// <summary>
        /// Computes the backpropagation to accumulate gradients.
        /// </summary>
        /// <returns>The gradient.</returns>
        [ConditionallyInternalUseOnly]
        public Tensor Back()
        {
            if (this.UpstreamGradient == null)
            {
                throw new InvalidOperationException("UpstreamGradient must be set before calling Back().");
            }

            if (this.IsDependentBranch)
            {
                var attribute = (ConditionallyInternalUseOnlyAttribute)Attribute.GetCustomAttribute(
                    MethodBase.GetCurrentMethod(), typeof(ConditionallyInternalUseOnlyAttribute));

                attribute?.Validate(this.IsDependentBranch, "Back should not be called on a dependent branch.");
            }

            Tensor currentUpstream = this.UpstreamGradient;

            // Reverse iterate over backpropagation steps to accumulate gradients
            foreach (var (step, result) in this.backpropagationSteps.AsEnumerable().Reverse())
            {
                // First, backpropagate through all branches
                foreach (var branch in result.Branches)
                {
                    var branchGradient = branch.Back();
                    currentUpstream = currentUpstream.ElementwiseAdd(branchGradient);
                }

                var (gradients, ops) = step(currentUpstream);
                currentUpstream = gradients[0];
                Parallel.For(0, result.Gradients.Length, i =>
                {
                    result.Gradients[i] = result.Gradients[i].ElementwiseAdd(gradients[i]);
                    if (ops[i] != null && !ops[i]!.IsDependentBranch)
                    {
                        ops[i]?.SetUpstreamGradient(gradients[i]);
                        ops[i]?.Back();
                    }
                });
            }

            this.SeedGradient = currentUpstream;
            return currentUpstream;
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
                var gradients = step(currentUpstream).Item1;
                currentUpstream = gradients[0];
                for (int i = 0; i < result.Gradients.Length; i++)
                {
                    result.Gradients[i] = result.Gradients[i].ElementwiseAdd(gradients[i]);
                }
            }
        }

        /// <summary>
        /// Gets the current tensor.
        /// </summary>
        /// <returns>The current tensor.</returns>
        internal Tensor GetCurrentTensor()
        {
            return this.currentTensor;
        }

        /// <summary>
        /// Gets the operation delegate for the specified operation.
        /// </summary>
        /// <typeparam name="T">The type of the delegate.</typeparam>
        /// <param name="operation">The operation func.</param>
        /// <returns>The operation delegate.</returns>
        internal T GetOperation<T>(Delegate operation)
            where T : Delegate
        {
            if (this.operations.TryGetValue(operation, out var instanceOperation))
            {
                return (T)instanceOperation;
            }

            throw new KeyNotFoundException("Operation not found.");
        }

        /// <summary>
        /// Initializes the operations dictionary using reflection.
        /// </summary>
        private void InitializeOperations()
        {
            var methods = this.GetType().GetMethods(BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public)
                .Where(m => m.GetCustomAttribute<PradOperationAttribute>() != null);

            foreach (var method in methods)
            {
                var attr = method.GetCustomAttribute<PradOperationAttribute>();
                var instanceDelegate = Delegate.CreateDelegate(attr.DelegateType, this, method);
                this.operations[attr.StaticDelegate] = instanceDelegate;
            }
        }
    }
}

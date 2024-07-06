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
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A lightweight reverse-mode automatic differentiation library.
    /// </summary>
    public class PradOp
    {
        private static PradOp funcOp = new PradOp();
        private readonly Tensor seed;
        private readonly Dictionary<Delegate, Delegate> operations;
        private Lazy<Ops> ops = new Lazy<Ops>(() => new Ops(), true);
        private Lazy<LossOps> lossOps = new Lazy<LossOps>(() => new LossOps(), true);
        private List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)> backpropagationSteps;
        private (Func<Tensor[], Tensor> splitStep, PradSplitResult result)? splitStep;
        private Tensor currentTensor;
        private PradResultBase parentResult;
        private PradResultBase initialResult;
        private Tensor seedGradient;
        private PradOp[] splitOps;
        private Guid id;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        /// <param name="seed">The seed tensor.</param>
        public PradOp(Tensor seed)
        {
            this.id = Guid.NewGuid();
            this.seed = seed;
            this.currentTensor = seed;
            this.seedGradient = new Tensor(seed.Shape);
            this.initialResult = new PradResult(this, seed, new Tensor[] { this.seedGradient });
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
        /// A custom tensor operation.
        /// </summary>
        /// <param name="tensors">The tensors as inputs.</param>
        /// <returns>The PradOp instance created in the function.</returns>
        public delegate PradOp TensorOp(params Tensor[] tensors);

        /// <summary>
        /// Gets the square root op.
        /// </summary>
        public static Func<PradResult> SquareRootOp => FuncOp.SquareRoot;

        /// <summary>
        /// Gets the add op.
        /// </summary>
        public static Func<Tensor, PradResult> AddOp => FuncOp.Add;

        /// <summary>
        /// Gets the mat mul op.
        /// </summary>
        public static Func<Tensor, PradResult> MatMulOp => FuncOp.MatMul;

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
        /// Gets the sub from op.
        /// </summary>
        public static Func<Tensor, PradResult> SubFromOp => FuncOp.SubFrom;

        /// <summary>
        /// Gets the div into op.
        /// </summary>
        public static Func<Tensor, PradResult> DivIntoOp => FuncOp.DivInto;

        /// <summary>
        /// Gets the expand dims op.
        /// </summary>
        public static Func<int, PradResult> ExpandDimsOp => FuncOp.ExpandDims;

        /// <summary>
        /// Gets the custom op.
        /// </summary>
        public static Func<TensorOp, Tensor[], PradResult> CustomTensorOp => FuncOp.CustomOperation;

        /// <summary>
        /// Gets the custom op.
        /// </summary>
        public static Func<Func<Tensor, Tensor>, Func<Tensor, Tensor, Tensor, Tensor[]>, PradResult> CustomOp => FuncOp.CustomOperation;

        /// <summary>
        /// Gets the sin op.
        /// </summary>
        public static Func<PradResult> SinOp => FuncOp.Sin;

        /// <summary>
        /// Gets the cos op.
        /// </summary>
        public static Func<PradResult> CosOp => FuncOp.Cos;

        /// <summary>
        /// Gets the reciprocal op.
        /// </summary>
        public static Func<PradResult> ReciprocalOp => FuncOp.Reciprocal;

        /// <summary>
        /// Gets the abs op.
        /// </summary>
        public static Func<PradResult> AbsOp => FuncOp.Abs;

        /// <summary>
        /// Gets the exp op.
        /// </summary>
        public static Func<PradResult> ExpOp => FuncOp.Exp;

        /// <summary>
        /// Gets the ln op.
        /// </summary>
        public static Func<PradResult> LnOp => FuncOp.Ln;

        /// <summary>
        /// Gets the log base 10 op.
        /// </summary>
        public static Func<PradResult> LogOp => FuncOp.Log;

        /// <summary>
        /// Gets the Gather op.
        /// </summary>
        public static Func<Tensor, int, PradResult> GatherOp => FuncOp.Gather;

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
        /// Gets the indexer op.
        /// </summary>
        public static Func<string[], PradResult> IndexerOp => FuncOp.Indexer;

        /// <summary>
        /// Gets the reshape op.
        /// </summary>
        public static Func<int[], PradResult> ReshapeOp => FuncOp.Reshape;

        /// <summary>
        /// Gets the transpose op.
        /// </summary>
        public static Func<int[], PradResult> TransposeOp => FuncOp.Transpose;

        /// <summary>
        /// Gets the tile op.
        /// </summary>
        public static Func<int[], PradResult> TileOp => FuncOp.Tile;

        /// <summary>
        /// Gets the clip op.
        /// </summary>
        public static Func<double, double, PradResult> ClipOp => FuncOp.Clip;

        /// <summary>
        /// Gets the exclude op.
        /// </summary>
        public static Func<double, double, PradResult> ExcludeOp => FuncOp.Exclude;

        /// <summary>
        /// Gets the sum op.
        /// </summary>
        public static Func<int[], PradResult> SumOp => FuncOp.Sum;

        /// <summary>
        /// Gets the broadcast to op.
        /// </summary>
        public static Func<int[], PradResult> BroadcastToOp => FuncOp.BroadcastTo;

        /// <summary>
        /// Gets the mean op.
        /// </summary>
        public static Func<int, PradResult> MeanOp => FuncOp.Mean;

        /// <summary>
        /// Gets the slice multiple 3D and concat op.
        /// </summary>
        public static Func<Tensor[], int, int[], PradResult> SliceMultiple3DAndConcatOp => FuncOp.SliceMultiple3DAndConcat;

        /// <summary>
        /// Gets operation types.
        /// </summary>
        public Ops Ops => this.ops.Value;

        /// <summary>
        /// Gets loss operation types.
        /// </summary>
        public LossOps LossOps => this.lossOps.Value;

        /// <summary>
        /// Gets the upstream gradient.
        /// </summary>
        public Tensor UpstreamGradient { get; private set; }

        /// <summary>
        /// Gets the seed gradient.
        /// </summary>
        public Tensor SeedGradient { get => this.initialResult.Gradients[0]; internal set => this.initialResult.Gradients[0].ReplaceData(value.Data); }

        /// <summary>
        /// Gets the seed result.
        /// </summary>
        public PradResult SeedResult => this.initialResult as PradResult ?? throw new InvalidCastException("Initial result is not a PradResult");

        /// <summary>
        /// Gets a value indicating whether this is a dependent branch. If so, Back should not be called.
        /// </summary>
        public bool IsDependentBranch => this.parentResult != null;

        /// <summary>
        /// Gets the shape of the current tensor.
        /// </summary>
        public int[] CurrentShape => this.currentTensor.Shape;

        /// <summary>
        /// Gets the ID.
        /// </summary>
        public Guid Id => this.id;

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor? Result
        {
            get
            {
                if (!this.backpropagationSteps.Any() && !this.IsDependentBranch)
                {
                    return default;
                }

                return new PradTensor(this, this.currentTensor);
            }
        }

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
        /// Print code for the current tensor.
        /// </summary>
        /// <returns>The C# code.</returns>
        public string PrintCodeForCurrentTensor()
        {
            return this.currentTensor.PrintCode();
        }

        /// <summary>
        /// Branch to another prad op.
        /// </summary>
        /// <returns>The other prad op.</returns>
        public PradOp Branch()
        {
            var branchedOp = new PradOp(this.currentTensor);
            if (this.backpropagationSteps.Any())
            {
                branchedOp.parentResult = this.backpropagationSteps.Last().result;
                branchedOp.parentResult.Branches.Add(branchedOp);
            }

            return branchedOp;
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
        /// <returns>The result of the custom operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(CustomOp))]
        public PradResult CustomOperation(
            Func<Tensor, Tensor> operation,
            Func<Tensor, Tensor, Tensor, Tensor[]> reverseOperation)
        {
            var input = this.currentTensor.DeepClone();

            // Apply the operation to the current tensor
            var result = operation(this.currentTensor);

            // Initialize gradient tensors
            var grad = Tensor.ToTensorArray(1, input.Shape);

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
        /// Applies a custom operation to the current tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="operation">The function that performs the operation.</param>
        /// <param name="otherTensors">The other tensors to use in the operation.</param>
        /// <returns>The result of the custom operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(CustomTensorOp))]
        public PradResult CustomOperation(TensorOp operation, params Tensor[] otherTensors)
        {
            var input = this.currentTensor.DeepClone();

            var inputData = this.currentTensor.Data;

            var allTensors = otherTensors.Prepend(this.currentTensor).ToArray();

            // Apply the operation to the current tensor
            var result = operation(allTensors);

            if (result.Result == null)
            {
                throw new InvalidOperationException("Result of custom operation is null.");
            }

            if (!result.SeedResult.ResultTensor.Data.SequenceEqual(inputData))
            {
                throw new InvalidOperationException("Seed tensor data does not match input tensor data.");
            }

            // Initialize gradient tensors
            var grad = Tensor.ToTensorArray(1, input.Shape);

            // Define the backpropagation step function
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var backResult = result.Back(upstreamGrad);
                var gradients = new Tensor[] { backResult };
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

            // Create the PradResult object with the result and gradient placeholders
            var pradResult = new PradResult(this, result.Result, grad);

            // Record the backpropagation step
            this.backpropagationSteps.Add((backpropStep, pradResult));

            // Update the current tensor
            this.currentTensor = result.Result;

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
        /// Subtracts the current tensor from another tensor element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to subtract from.</param>
        /// <returns>The result of the subtraction along with the gradient placeholders.</returns>
        [PradOperation(nameof(SubFromOp))]
        public PradResult SubFrom(Tensor tensor)
        {
            var result = tensor.ElementwiseSub(this.currentTensor);
            var tensorReverse = new TensorReverse(new Tensor[] { tensor, new Tensor(this.currentTensor) });

            var grad = Tensor.ToTensorArray(2, tensor.Shape);
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

                return (new Tensor[] { gradients[1], gradients[0] }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs matrix multiplication and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to multiply with.</param>
        /// <returns>The result of the multiplication along with the gradient placeholders.</returns>
        [PradOperation(nameof(MatMulOp))]
        public PradResult MatMul(Tensor tensor)
        {
            var result = this.currentTensor.MatrixMultiply(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = new Tensor[] { new Tensor(this.currentTensor.Shape), new Tensor(tensor.Shape) };
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.MatrixMultiplyReverse(upstreamGrad);
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
            var tensorReverse = new TensorReverse(new Tensor[] { new Tensor(this.currentTensor) });

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
        /// Divides a tensor by the current tensor element-wise and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensor">The tensor to divide into.</param>
        /// <returns>The result of the division along with the gradient placeholders.</returns>
        [PradOperation(nameof(DivIntoOp))]
        public PradResult DivInto(Tensor tensor)
        {
            var denominator = new Tensor(this.currentTensor);
            var result = tensor.ElementwiseDivide(this.currentTensor);
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            var grad = Tensor.ToTensorArray(2, tensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ElementwiseDivideReverse(upstreamGrad, denominator);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { denominator, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (new Tensor[] { gradients[1], gradients[0] }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Expands the dimensions of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="axis">The axis along which to expand the dimensions.</param>
        /// <returns>The result of the expand dims operation.</returns>
        [PradOperation(nameof(ExpandDimsOp))]
        public PradResult ExpandDims(int axis = -1)
        {
            var result = this.currentTensor.ExpandDims(axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ExpandDimsReverse(upstreamGrad, axis);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Perform a no-op and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the no-op operation along with the gradient placeholders.</returns>
        public PradResult NoOp()
        {
            var resultTensor = new Tensor(this.currentTensor.Shape);
            var elementSize = PradTools.GetElementSize(this.currentTensor.Data);
            Buffer.BlockCopy(this.currentTensor.Data, 0, resultTensor.Data, 0, this.currentTensor.Data.Length * elementSize);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradientTensor = new Tensor(upstreamGrad.Shape);
                Buffer.BlockCopy(upstreamGrad.Data, 0, gradientTensor.Data, 0, upstreamGrad.Data.Length * elementSize);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradientTensor }, ops);
            };

            var pradResult = new PradResult(this, resultTensor, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = resultTensor;
            return pradResult;
        }

        /// <summary>
        /// Computes the reciprocal of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the reciprocal operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(ReciprocalOp))]
        public PradResult Reciprocal()
        {
            var result = this.currentTensor.Reciprocal();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ReciprocalReverse(upstreamGrad, result);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the abs of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the abs operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(AbsOp))]
        public PradResult Abs()
        {
            var result = this.currentTensor.Abs();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.AbsReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the exp of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the exp operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(ExpOp))]
        public PradResult Exp()
        {
            var result = this.currentTensor.Exp();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ExpReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the ln of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the exp operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(LnOp))]
        public PradResult Ln()
        {
            var result = this.currentTensor.Exp();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.LnReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the log base 10 of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the log base 10 operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(LogOp))]
        public PradResult Log()
        {
            var result = this.currentTensor.Log();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.LogReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
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
        [PradOperation(nameof(IndexerOp))]
        public PradResult Indexer(params string[] indices)
        {
            var result = this.currentTensor.Indexer(indices);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.IndexerReverse(upstreamGrad, indices);
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
        [PradOperation(nameof(ReshapeOp))]
        public PradResult Reshape(params int[] newShape)
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
        [PradOperation(nameof(TransposeOp))]
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

            var pradResult = new PradResult(this, results[0], grad);
            var splitResult = new PradSplitResult(results, grad);
            splitResult.PradOp = this;
            var split = (splitStep, splitResult);
            var currentTensor = results[0];
            var newOp = new PradOp(currentTensor);

            this.splitStep = split;

            var branch = newOp.SplitBranch();
            var splitOps = new PradOp[] { newOp, branch };
            this.splitOps = splitOps;
            return splitOps;
        }

        /// <summary>
        /// Broadcasts the tensor to a specified shape.
        /// </summary>
        /// <param name="newShape">The new shape to broadcast to.</param>
        /// <returns>A new tensor broadcasted to the specified shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the new shape is not compatible for broadcasting.</exception>
        [PradOperation(nameof(BroadcastToOp))]
        public PradResult BroadcastTo(params int[] newShape)
        {
            var result = this.currentTensor.BroadcastTo(newShape);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.BroadcastToReverse(upstreamGrad, newShape);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Sums the tensor elements along the specified axes.
        /// </summary>
        /// <param name="axes">The axes along which to sum the elements.</param>
        /// <returns>A new tensor with the summed elements.</returns>
        [PradOperation(nameof(SumOp))]
        public PradResult Sum(params int[] axes)
        {
            var result = this.currentTensor.Sum(axes);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SumReverse(upstreamGrad, axes);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Excludes the values of the tensor that are in the specified range.
        /// If values within the range are closer to max, they become max, otherwise they become min.
        /// </summary>
        /// <param name="min">The min value to exclude.</param>
        /// <param name="max">The max value to exclude.</param>
        /// <returns>A new tensor with values in the specified range excluded.</returns>
        [PradOperation(nameof(ExcludeOp))]
        public PradResult Exclude(double min, double max)
        {
            var result = this.currentTensor.Exclude(min, max);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ExcludeReverse(upstreamGrad, min, max);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Clips the values of the tensor to be within the specified range.
        /// </summary>
        /// <param name="min">The minimum value to clip to.</param>
        /// <param name="max">The maximum value to clip to.</param>
        /// <returns>A new tensor with values clipped to the specified range.</returns>
        [PradOperation(nameof(ClipOp))]
        public PradResult Clip(double min, double max)
        {
            var result = this.currentTensor.Clip(min, max);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ClipReverse(upstreamGrad, min, max);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Tiles the tensor along each dimension and records the operation for backpropagation.
        /// </summary>
        /// <param name="multiples">The array of multiples for each dimension.</param>
        /// <returns>The tiled tensor along with the gradient placeholders.</returns>
        [PradOperation(nameof(TileOp))]
        public PradResult Tile(params int[] multiples)
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
        [PradOperation(nameof(GatherOp))]
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
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

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
        /// Computes the mean of the tensor along the specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="axis">The axis along which to compute the mean.</param>
        /// <returns>The result of the mean operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(MeanOp))]
        public PradResult Mean(int axis)
        {
            var result = this.currentTensor.Mean(axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.MeanReverse(upstreamGrad, axis);
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
        /// Slice multiple 3-D tensors and concatenate along the specified axis.
        /// </summary>
        /// <param name="tensors">The 3-D tensors to concatenate.</param>
        /// <param name="axis">The axis to concatenate on.</param>
        /// <param name="sliceSizes">The slice size to extract.</param>
        /// <returns>A PradResult.</returns>
        [PradOperation(nameof(SliceMultiple3DAndConcatOp))]
        public PradResult SliceMultiple3DAndConcat(Tensor[] tensors, int axis, params int[] sliceSizes)
        {
            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);
            var result = Tensor.Slice3DTensors(tensors, sliceSizes);
            var resultArray = result.ToArray();
            var concatResult = Tensor.Concat(resultArray, axis);
            var concatTensorReverse = new TensorReverse(resultArray);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var concatGradients = concatTensorReverse.ConcatReverse(upstreamGrad, axis);
                var gradients = TensorReverse.Slice3DTensorsReverse(concatGradients, combinedTensors, sliceSizes);
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

            var pradResult = new PradResult(this, concatResult, grads);
            this.backpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = concatResult;
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
                pradOps[i] = new PradOp(this.currentTensor);
            }

            var results = new PradResult[operations.Length];

            // Use Parallel.For to execute operations in parallel
            Parallel.For(0, operations.Length, i =>
            {
                results[i] = operations[i](pradOps[i]);
            });

            for (int i = 1; i < operations.Length; i++)
            {
                pradOps[i].RecordSplitBranch(pradOps[0]);
            }

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

            if (this.splitOps != null)
            {
                var gradientLeft = this.splitOps[0].Back(this.UpstreamGradient);
                var gradientRight = this.splitOps[1].Back();

                if (this.splitStep.HasValue)
                {
                    this.UpstreamGradient = this.splitStep.Value.splitStep(new Tensor[] { gradientLeft, gradientRight });
                }
            }

            Tensor currentUpstream = this.UpstreamGradient;

            // Reverse iterate over backpropagation steps to accumulate gradients
            foreach (var (step, result) in this.backpropagationSteps.AsEnumerable().Reverse())
            {
                // First, backpropagate through all branches
                foreach (var branch in result.Branches.Where(x => x.UpstreamGradient != null))
                {
                    var branchGradient = branch.Back();
                    currentUpstream = currentUpstream.ElementwiseAdd(branchGradient);
                }

                // Then, backpropagate through all split branches
                List<Tensor> branchGradients = new List<Tensor>();
                foreach (var branch in result.SplitBranches)
                {
                    var branchGradient = branch.Back();
                    branchGradients.Add(branchGradient);
                }

                var (gradients, ops) = step(currentUpstream);

                foreach (var branchGradient in branchGradients)
                {
                    gradients[0] = gradients[0].ElementwiseAdd(branchGradient);
                }

                currentUpstream = gradients[0];

                Parallel.For(0, result.Gradients.Length, i =>
                {
                    result.Gradients[i] = result.Gradients[i].ElementwiseAdd(gradients[i]);
                    if (ops[i] != null)
                    {
                        ops[i]?.SetUpstreamGradient(gradients[i]);
                        if (!ops[i]!.IsDependentBranch)
                        {
                            ops[i]?.Back();
                        }
                    }
                });
            }

            foreach (var branch in this.initialResult.Branches)
            {
                var branchGradient = branch.Back();
                currentUpstream = currentUpstream.ElementwiseAdd(branchGradient);
            }

            this.SeedGradient = currentUpstream;
            return currentUpstream;
        }

        /// <summary>
        /// Is the result currently associated with the PradOp instance.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <returns>A value indicating whether the PradOp instance for the result has not been modified.</returns>
        internal bool IsCurrentlyAssociated(PradResult result)
        {
            return this.backpropagationSteps.Any() && this.backpropagationSteps.Last().result == result;
        }

        /// <summary>
        /// Branch to another prad op after the fact.
        /// </summary>
        /// <param name="parentResult">The parent result.</param>
        /// <returns>The other prad op.</returns>
        internal PradOp BranchAfterTheFact(PradResult parentResult)
        {
            var branchedOp = new PradOp(parentResult.ResultTensor);
            if (this.backpropagationSteps.Any())
            {
                branchedOp.parentResult = parentResult;
                branchedOp.parentResult.Branches.Add(branchedOp);
            }

            return branchedOp;
        }

        /// <summary>
        /// Branch to another prad op.
        /// </summary>
        /// <returns>The other prad op.</returns>
        internal PradOp SplitBranch()
        {
            var branchedOp = new PradOp(this.currentTensor);
            if (this.backpropagationSteps.Any())
            {
                branchedOp.parentResult = this.backpropagationSteps.Last().result;
                branchedOp.parentResult.SplitBranches.Add(branchedOp);
            }
            else if (this.splitStep.HasValue)
            {
                branchedOp.parentResult = this.splitStep.Value.result;
            }

            return branchedOp;
        }

        /// <summary>
        /// Branch to another prad op.
        /// </summary>
        /// <param name="originalOp">The original prad op.</param>
        /// <returns>The other prad op.</returns>
        internal PradOp RecordSplitBranch(PradOp originalOp)
        {
            if (this.backpropagationSteps.Any())
            {
                var parentResult = originalOp.backpropagationSteps.Last().result;
                this.parentResult = parentResult;
                parentResult.SplitBranches.Add(this);
            }

            return this;
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

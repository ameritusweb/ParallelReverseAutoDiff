﻿//------------------------------------------------------------------------------
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
    using static ParallelReverseAutoDiff.PRAD.PradOp;

    /// <summary>
    /// A lightweight reverse-mode automatic differentiation library.
    /// </summary>
    public partial class PradOp
    {
        private static PradOp funcOp = new PradOp();
        private readonly Dictionary<Delegate, Delegate> operations;
        private Tensor seed;
        private (Func<Tensor[], Tensor> splitStep, PradSplitResult result)? splitStep;
        private Tensor currentTensor;
        private PradResultBase parentResult;
        private PradResultBase initialResult;
        private Tensor seedGradient;
        private PradOp[]? splitOps;
        private Guid id;
        private PradOp originalSplitOp; // Link to the original PradOp that did the split
        private int splitIndex; // Index of this PradOp in the split
        private int gradientStackCounter; // Counter to track the number of gradients received
        private Tensor[] splitGradients; // Array to store gradients from each split
        private PradOpBranchTracker branchTracker;
        private IList<Tensor> seedGradientHistory;
        private Guid engineId;
        private Dictionary<Guid, List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>> backpropagationStepsByEngine =
            new Dictionary<Guid, List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>>();

        private Tensor lastSeedGradient;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradOp"/> class.
        /// </summary>
        /// <param name="seed">The seed tensor.</param>
        public PradOp(Tensor seed)
        {
            this.id = Guid.NewGuid();
            this.engineId = Guid.NewGuid();
            this.seed = seed;
            this.currentTensor = seed;
            this.seedGradient = new Tensor(seed.Shape);
            this.initialResult = new PradResult(this, seed, new Tensor[] { this.seedGradient });
            this.backpropagationStepsByEngine = new Dictionary<Guid, List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>>();
            this.operations = new Dictionary<Delegate, Delegate>();
            this.branchTracker = new PradOpBranchTracker();
            this.seedGradientHistory = new List<Tensor>();
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
        /// Gets the embedding op.
        /// </summary>
        public static Func<Tensor, PradResult> EmbeddingOp => FuncOp.Embedding;

        /// <summary>
        /// Gets the sub from op.
        /// </summary>
        public static Func<Tensor, PradResult> SubFromOp => FuncOp.SubFrom;

        /// <summary>
        /// Gets the div into op.
        /// </summary>
        public static Func<Tensor, PradResult> DivIntoOp => FuncOp.DivInto;

        /// <summary>
        /// Gets the modulus op.
        /// </summary>
        public static Func<Tensor, PradResult> ModulusOp => FuncOp.Modulus;

        /// <summary>
        /// Gets the expand dims op.
        /// </summary>
        public static Func<int, PradResult> ExpandDimsOp => FuncOp.ExpandDims;

        /// <summary>
        /// Gets the argmax op.
        /// </summary>
        public static Func<int, PradResult> ArgMaxOp => FuncOp.ArgMax;

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
        /// Gets the extract patches op.
        /// </summary>
        public static Func<int[], int[], string, PradResult> ExtractPatchesOp => FuncOp.ExtractPatches;

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
        /// Gets the BesselI0 op.
        /// </summary>
        public static Func<PradResult> BesselI0Op => FuncOp.BesselI0;

        /// <summary>
        /// Gets the ln op.
        /// </summary>
        public static Func<PradResult> LnOp => FuncOp.Ln;

        /// <summary>
        /// Gets the log base 10 op.
        /// </summary>
        public static Func<PradResult> LogOp => FuncOp.Log;

        /// <summary>
        /// Gets the tanh op.
        /// </summary>
        public static Func<PradResult> TanhOp => FuncOp.Tanh;

        /// <summary>
        /// Gets the diff op.
        /// </summary>
        public static Func<int, PradResult> DiffOp => FuncOp.Diff;

        /// <summary>
        /// Gets the sigmoid op.
        /// </summary>
        public static Func<PradResult> SigmoidOp => FuncOp.Sigmoid;

        /// <summary>
        /// Gets the leaky ReLU op.
        /// </summary>
        public static Func<PradResult> LeakyReLUOp => FuncOp.LeakyReLU;

        /// <summary>
        /// Gets the Gather op.
        /// </summary>
        public static Func<Tensor, int, PradResult> GatherOp => FuncOp.Gather;

        /// <summary>
        /// Gets the Interleaved Gather op.
        /// </summary>
        public static Func<int, int, PradResult> InterleavedGatherOp => FuncOp.InterleavedGather;

        /// <summary>
        /// Gets the Interleaved Gather Inverse op.
        /// </summary>
        public static Func<int, int, PradResult> InterleavedGatherInverseOp => FuncOp.InterleavedGatherInverse;

        /// <summary>
        /// Gets the GatherNd op.
        /// </summary>
        public static Func<Tensor, PradResult> GatherNdOp => FuncOp.GatherNd;

        /// <summary>
        /// Gets the sum rows op.
        /// </summary>
        public static Func<PradResult> SumRowsOp => FuncOp.SumRows;

        /// <summary>
        /// Gets the self pair op.
        /// </summary>
        public static Func<PradResult> SelfPairOp => FuncOp.SelfPair;

        /// <summary>
        /// Gets the multiply columns op.
        /// </summary>
        public static Func<PradResult> MultiplyColumnsOp => FuncOp.MultiplyColumns;

        /// <summary>
        /// Gets the square op.
        /// </summary>
        public static Func<PradResult> SquareOp => FuncOp.Square;

        /// <summary>
        /// Gets the arccos op.
        /// </summary>
        public static Func<PradResult> ArcCosOp => FuncOp.ArcCos;

        /// <summary>
        /// Gets the atan2 op.
        /// </summary>
        public static Func<Tensor, PradResult> Atan2Op => FuncOp.Atan2;

        /// <summary>
        /// Gets the max op.
        /// </summary>
        public static Func<Tensor, PradResult> MaxOp => FuncOp.Max;

        /// <summary>
        /// Gets the pow op.
        /// </summary>
        public static Func<object, PradResult> PowOp => FuncOp.Pow;

        /// <summary>
        /// Gets the min op.
        /// </summary>
        public static Func<Tensor, PradResult> MinOp => FuncOp.Min;

        /// <summary>
        /// Gets the on-off embedding op.
        /// </summary>
        public static Func<Tensor, Tensor, Tensor, PradResult> OnOffEmbeddingOp => FuncOp.OnOffEmbedding;

        /// <summary>
        /// Gets the equals op.
        /// </summary>
        public static Func<Tensor, PradResult> EqualsOp => FuncOp.Equals;

        /// <summary>
        /// Gets the less than op.
        /// </summary>
        public static Func<Tensor, PradResult> LessThanOp => FuncOp.LessThan;

        /// <summary>
        /// Gets the greater than op.
        /// </summary>
        public static Func<Tensor, PradResult> GreaterThanOp => FuncOp.GreaterThan;

        /// <summary>
        /// Gets the pairwise tile op.
        /// </summary>
        public static Func<Tensor, PradResult> PairwiseTileOp => FuncOp.PairwiseTile;

        /// <summary>
        /// Gets the where op.
        /// </summary>
        public static Func<Tensor, Tensor, PradResult> WhereOp => FuncOp.Where;

        /// <summary>
        /// Gets the stack op.
        /// </summary>
        public static Func<Tensor[], int, PradResult> StackOp => FuncOp.Stack;

        /// <summary>
        /// Gets the concat op.
        /// </summary>
        public static Func<Tensor[], int, int[], PradResult> ConcatOp => FuncOp.Concat;

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
        /// Gets the upstream gradient.
        /// </summary>
        public Tensor UpstreamGradient { get; private set; }

        /// <summary>
        /// Gets a value indicating whether or not the branch has started.
        /// </summary>
        public bool IsStarted { get; private set; }

        /// <summary>
        /// Gets a value indicating whether or not the branch is finished.
        /// </summary>
        public bool IsFinished { get; private set; }

        /// <summary>
        /// Gets the start date.
        /// </summary>
        public DateTimeOffset StartDate { get; private set; }

        /// <summary>
        /// Gets the finish date.
        /// </summary>
        public DateTimeOffset FinishDate { get; private set; }

        /// <summary>
        /// Gets the reset date.
        /// </summary>
        public DateTimeOffset ResetDate { get; private set; }

        /// <summary>
        /// Gets or sets the backpropagation mode.
        /// </summary>
        public BackpropagationMode BackpropagationMode { get; set; }

        /// <summary>
        /// Gets the seed gradient.
        /// </summary>
        public Tensor SeedGradient => this.lastSeedGradient;

        /// <summary>
        /// Gets the seed result.
        /// </summary>
        public PradResult SeedResult => this.initialResult as PradResult ?? throw new InvalidCastException("Initial result is not a PradResult");

        /// <summary>
        /// Gets the branch initial tensor.
        /// </summary>
        public Tensor BranchInitialTensor => this.SeedResult.Result;

        /// <summary>
        /// Gets the tensor result of the computation if a computation occurred, or the initial seed tensor if a computation has not occurred.
        /// </summary>
        public Tensor CurrentTensor
        {
            get
            {
                return this.Result ?? this.BranchInitialTensor;
            }
        }

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
        /// Gets or sets the linked branches.
        /// </summary>
        public List<PradOp> LinkedBranches { get; set; } = new List<PradOp>();

        /// <summary>
        /// Gets or sets the tensors waiting for backpropagation to finish.
        /// </summary>
        public Queue<Tensor> WaitingToAdd { get; set; } = new Queue<Tensor>();

        /// <summary>
        /// Gets or sets a value indicating whether the operation has a low backpropagation priority.
        /// </summary>
        public bool IsLowPriorityForBackpropagation { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether the operation has a high backpropagation priority.
        /// </summary>
        public bool IsHighPriorityForBackpropagation { get; set; } = false;

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor? Result
        {
            get
            {
                if (!this.BackpropagationSteps.Any() && !this.IsDependentBranch)
                {
                    return default;
                }

                return new PradTensor(this, this.currentTensor);
            }
        }

        /// <summary>
        /// Gets or sets the current engine ID.
        /// </summary>
        public Guid EngineId
        {
            get => this.engineId;
            set
            {
                this.engineId = value;
                this.IsStarted = false;
                this.IsFinished = false;
                this.UpstreamGradient = null!;
            }
        }

        /// <summary>
        /// Gets the average gradient.
        /// </summary>
        public Tensor? AverageGradient
        {
            get
            {
                if (this.BackpropagationMode == BackpropagationMode.Replace
                    ||
                    !this.seedGradientHistory.Any())
                {
                    return default;
                }

                var firstSeed = this.seedGradientHistory.First();
                foreach (var seed in this.seedGradientHistory.Skip(1))
                {
                    firstSeed = firstSeed.ElementwiseAdd(seed);
                }

                var avg = firstSeed.ElementwiseDivide(new Tensor(firstSeed.Shape, this.seedGradientHistory.Count));
                return avg;
            }
        }

        /// <summary>
        /// Gets an operation to get a func.
        /// </summary>
        internal static PradOp FuncOp => funcOp;

        /// <summary>
        /// Gets or sets the last gradient.
        /// </summary>
        internal Tensor LastGradient { get => this.initialResult.Gradients[0]; set => this.initialResult.Gradients[0].ReplaceData(value.Data); }

        /// <summary>
        /// Gets or sets the current backpropagation steps.
        /// </summary>
        internal List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)> BackpropagationSteps
        {
            get
            {
                if (!this.backpropagationStepsByEngine.ContainsKey(this.engineId))
                {
                    this.backpropagationStepsByEngine[this.engineId] = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();
                }

                return this.backpropagationStepsByEngine[this.engineId];
            }

            set
            {
                this.backpropagationStepsByEngine[this.engineId] = value;
            }
        }

        /// <summary>
        /// Gets the last result.
        /// </summary>
        internal PradResult? LastResult
        {
            get
            {
                if (this.BackpropagationSteps.Any())
                {
                    return this.BackpropagationSteps.Last().result;
                }

                return default;
            }
        }

        /// <summary>
        /// Resets backpropagation so that you can call 'Back' with a different upstream gradient.
        /// </summary>
        public void ResetBackpropagation()
        {
            if (!this.IsFinished)
            {
                return;
            }

            // Store the last gradient before reset
            this.lastSeedGradient = this.LastGradient?.DeepClone() !;

            // Existing reset logic
            this.IsStarted = false;
            this.IsFinished = false;
            this.ResetDate = DateTimeOffset.UtcNow;
            this.UpstreamGradient = null !;

            // Reset gradient states
            this.seedGradient = new Tensor(this.seed.Shape);
            this.initialResult = new PradResult(this, this.seed, new Tensor[] { this.seedGradient });

            // Cascade through linked branches
            foreach (var branch in this.LinkedBranches)
            {
                branch.ResetBackpropagation();
            }

            if (this.parentResult != null)
            {
                foreach (var branch in this.parentResult.Branches)
                {
                    branch.ResetBackpropagation();
                }
            }

            // Cascade through computation graph
            foreach (var step in this.BackpropagationSteps)
            {
                foreach (var branch in step.result.Branches)
                {
                    branch.ResetBackpropagation();  // Regular branches
                }

                foreach (var branch in step.result.SplitBranches)
                {
                    branch.ResetBackpropagation();  // Split branches
                }
            }

            // Cascade through split ops
            if (this.splitOps != null)
            {
                foreach (var splitOp in this.splitOps)
                {
                    splitOp.ResetBackpropagation();  // Split operations
                }
            }
        }

        /// <summary>
        /// Gets the steps for a certain engine ID.
        /// </summary>
        /// <param name="engineId">The engine ID.</param>
        /// <returns>The backpropagation steps.</returns>
        public List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)> GetBackpropagationSteps(Guid engineId)
        {
            if (!this.backpropagationStepsByEngine.ContainsKey(engineId))
            {
                this.backpropagationStepsByEngine[engineId] = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();
            }

            return this.backpropagationStepsByEngine[engineId];
        }

        /// <summary>
        /// Sets the branch tracker.
        /// </summary>
        /// <param name="branchTracker">The branch tracker.</param>
        public void SetBranchTracker(PradOpBranchTracker branchTracker)
        {
            this.branchTracker = branchTracker;
        }

        /// <summary>
        /// Resets the gradient.
        /// </summary>
        public void ResetGradient()
        {
            this.seedGradient = new Tensor(this.seed.Shape);
            this.initialResult = new PradResult(this, this.seed, new Tensor[] { this.seedGradient });
            this.seedGradientHistory.Clear();
        }

        /// <summary>
        /// Resets both the gradient and the backpropagation steps.
        /// </summary>
        public void Reset()
        {
            // Reset the gradient
            this.ResetGradient();

            this.backpropagationStepsByEngine.Clear();

            // Reset the backpropagation steps
            this.EngineId = Guid.NewGuid();
            this.backpropagationStepsByEngine[this.engineId] = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();

            this.LinkedBranches.Clear();
            this.branchTracker.VisitedBranches.Clear();

            // Reset current tensor to seed
            this.currentTensor = this.seed;
        }

        /// <summary>
        /// Resets both the gradient and the backpropagation steps.
        /// </summary>
        /// <param name="tensor">The tensor to reset with.</param>
        public void Reset(Tensor tensor)
        {
            // Reset the gradient
            this.ResetGradient();

            this.backpropagationStepsByEngine.Clear();

            // Reset the backpropagation steps
            this.EngineId = Guid.NewGuid();
            this.backpropagationStepsByEngine[this.engineId] = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();

            this.LinkedBranches.Clear();
            this.branchTracker.VisitedBranches.Clear();

            // Reset current tensor to seed
            this.currentTensor = tensor;
            this.seed = tensor;
            this.seedGradient = new Tensor(this.seed.Shape);
            this.initialResult = new PradResult(this, this.seed, new Tensor[] { this.seedGradient });
        }

        /// <summary>
        /// Sets the upstream gradient.
        /// </summary>
        /// <param name="gradient">The upstream gradient.</param>
        /// <exception cref="ArgumentException">Shapes not matching.</exception>
        public void SetUpstreamGradient(Tensor gradient)
        {
            if (this.UpstreamGradient != null)
            {
                if (!gradient.Shape.SequenceEqual(this.UpstreamGradient.Shape))
                {
                    throw new ArgumentException("Upstream gradient shape must match the current tensor shape.");
                }

                this.UpstreamGradient = this.UpstreamGradient.ElementwiseAdd(gradient);
                return;
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
            if (!this.BackpropagationSteps.Any())
            {
                this.NoOp();
            }

            var branchedOp = new PradOp(this.currentTensor);
            branchedOp.SetBranchTracker(this.branchTracker);

            if (this.BackpropagationSteps.Any())
            {
                branchedOp.parentResult = this.BackpropagationSteps.Last().result;
                branchedOp.parentResult.Branches.Add(branchedOp);
            }

            return branchedOp;
        }

        /// <summary>
        /// Takes back a branch.
        /// </summary>
        public void TakeBackBranch()
        {
            if (this.parentResult.Branches.Contains(this))
            {
                this.parentResult.Branches.Remove(this);
            }
        }

        /// <summary>
        /// Creates a stack of branches from the current PradOp instance.
        /// </summary>
        /// <param name="n">The number of branches to create.</param>
        /// <returns>A BranchStack containing the branches.</returns>
        public BranchStack BranchStack(int n)
        {
            if (n < 1)
            {
                throw new ArgumentException("Number of branches must be at least 1.", nameof(n));
            }

            var branches = new PradOp[n];

            for (int i = 0; i < n; i++)
            {
                branches[i] = this.Branch(); // Create additional branches
            }

            return new BranchStack(branches);
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
            clonedOp.BackpropagationSteps = new List<(Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)>();
            foreach (var (backpropStep, result) in this.BackpropagationSteps)
            {
                var clonedResult = new PradResult(
                    clonedOp,
                    result.Result.DeepClone(),
                    result.Gradients.Select(g => g.DeepClone()).ToArray());
                clonedOp.BackpropagationSteps.Add((backpropStep, clonedResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));

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
                result.UpstreamGradient = upstreamGrad;
                var backResult = result.Back();
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));

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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs an embedding operation.
        /// </summary>
        /// <param name="embeddings">The embeddings.</param>
        /// <returns>The embedded tensor.</returns>
        [PradOperation(nameof(EmbeddingOp))]
        public PradResult Embedding(Tensor embeddings)
        {
            var originalShape = this.currentTensor.Shape;
            var result = this.currentTensor.Embedding(embeddings);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, embeddings });

            var grad = new Tensor[] { new Tensor(originalShape), new Tensor(embeddings.Shape) };
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.EmbeddingReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, embeddings };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (new Tensor[] { new Tensor(originalShape), gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

            var broadcastingRequired = !this.currentTensor.Shape.SequenceEqual(tensor.Shape);

            var result = broadcastingRequired ? default : this.currentTensor.ElementwiseMultiply(tensor);

            var bResult = broadcastingRequired ? this.currentTensor.ElementwiseMultiplyBroadcasting(tensor) : default;

            if (broadcastingRequired)
            {
                result = bResult!.Value.result;
            }

            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, tensor });

            var grad = new Tensor[] { new Tensor(this.currentTensor.Shape), new Tensor(tensor.Shape) };
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = !broadcastingRequired
                    ?
                    tensorReverse.ElementwiseMultiplyReverse(upstreamGrad)
                    :
                    tensorReverse.ElementwiseMultiplyBroadcastingReverse(upstreamGrad, bResult!.Value.mapping);
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

            var pradResult = new PradResult(this, result!, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result!;
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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the sigmoid of each element in the tensor.
        /// </summary>
        /// <returns>The result of the tanh operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(SigmoidOp))]
        public PradResult Sigmoid()
        {
            var result = this.currentTensor.Sigmoid();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SigmoidReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the hyperbolic tangent of each element in the tensor.
        /// </summary>
        /// <returns>The result of the tanh operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(TanhOp))]
        public PradResult Tanh()
        {
            var result = this.currentTensor.ElementwiseTanh();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseTanhReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the leaky ReLU of each element in the tensor.
        /// </summary>
        /// <returns>The result of the leaky ReLU operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(LeakyReLUOp))]
        public PradResult LeakyReLU()
        {
            var result = this.currentTensor.ElementwiseLeakyReLU(0.01d);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseLeakyReLUReverse(upstreamGrad, 0.01d);
                PradOp?[] ops = new PradOp?[1];
                return (new[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the first-order diff of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <param name="axis">The axis along which to compute the diff.</param>
        /// <returns>The result of the first-order diff operation.</returns>
        [PradOperation(nameof(DiffOp))]
        public PradResult Diff(int axis)
        {
            var result = this.currentTensor.Diff(axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.DiffReverse(upstreamGrad, axis);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Finds the indices of the maximum values along the specified axis with optimized memory access and vectorized operations using Vector of double.
        /// </summary>
        /// <param name="axis">The axis along which to find the maximum indices. If -1, finds the index of the maximum value in the flattened tensor.</param>
        /// <returns>A new tensor containing the indices of the maximum values.</returns>
        [PradOperation(nameof(ArgMaxOp))]
        public PradResult ArgMax(int axis = -1)
        {
            var result = this.currentTensor.ArgMax(axis);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = new Tensor(this.currentTensor.Shape);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = resultTensor;
            return pradResult;
        }

        /// <summary>
        /// Computes the reciprocal of each element of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The result of the reciprocal operation along with the gradient placeholders.</returns>
        [PradOperation(nameof(BesselI0Op))]
        public PradResult BesselI0()
        {
            var result = this.currentTensor.BesselI0();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.BesselI0Reverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Extracts patches from a 2D matrix tensor (using MKLNET for optimization).
        /// </summary>
        /// <param name="filterSize">The size of the sliding window [filter_height, filter_width].</param>
        /// <param name="strides">The strides for the sliding window [stride_height, stride_width].</param>
        /// <param name="padding">Padding type ('VALID' or 'SAME').</param>
        /// <returns>A new tensor containing the extracted patches.</returns>
        [PradOperation(nameof(ExtractPatchesOp))]
        public PradResult ExtractPatches(int[] filterSize, int[] strides, string padding)
        {
            var result = this.currentTensor.ExtractPatches(filterSize, strides, padding);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ExtractPatchesReverse(upstreamGrad, filterSize, strides, padding);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Generates unique pairs within a single 1D tensor without repetition or self-pairing.
        /// Optimized with precomputed offsets and parallelism.
        /// </summary>
        /// <returns>A result of shape [2, M] where each column represents a unique pair.</returns>
        [PradOperation(nameof(SelfPairOp))]
        public PradResult SelfPair()
        {
            var result = this.currentTensor.SelfPair();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.SelfPairReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs an element-wise maximum operation between this tensor and the provided tensor.
        /// </summary>
        /// <param name="other">The tensor to compare with the current tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> containing the result tensor of the element-wise maximum operation
        /// and the associated gradients for backpropagation.
        /// </returns>
        /// <remarks>
        /// This method computes the element-wise maximum of two tensors, stores the result in the current tensor,
        /// and registers a backpropagation step to calculate the reverse gradient during the backpropagation phase.
        /// </remarks>
        /// <example>
        /// <code>
        /// var tensor1 = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        /// var tensor2 = new Tensor(new int[] { 2, 2 }, new double[] { 2, 1, 4, 3 });
        /// var result = tensor1.Max(tensor2);
        /// // Result tensor: { 2, 2, 4, 4 }
        /// </code>
        /// </example>
        [PradOperation(nameof(MaxOp))]
        public PradResult Max(Tensor other)
        {
            var result = this.currentTensor.Max(other);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.MaxReverse(upstreamGrad, other);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, other };
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs an element-wise power operation on this tensor with the provided exponent.
        /// </summary>
        /// <param name="exponent">The exponent to raise each element to. Can be a scalar double or a Tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> containing the result tensor of the element-wise power operation
        /// and the associated gradients for backpropagation.
        /// </returns>
        /// <remarks>
        /// This method computes the element-wise power of the current tensor, stores the result,
        /// and registers a backpropagation step to calculate the reverse gradient during the backpropagation phase.
        /// </remarks>
        /// <example>
        /// <code>
        /// var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        /// var result = tensor.Pow(2.0);
        /// // Result tensor: { 1, 4, 9, 16 }
        /// =================================
        /// var exponentTensor = new Tensor(new int[] { 2, 2 }, new double[] { 2, 3, 2, 3 });
        /// var result2 = tensor.Pow(exponentTensor);
        /// // Result tensor: { 1, 8, 9, 64 }
        /// </code>
        /// </example>
        [PradOperation(nameof(PowOp))]
        public PradResult Pow(object exponent)
        {
            Tensor result;
            Tensor? exponentTensor = null;
            if (exponent is float scalarExponentF)
            {
                result = this.currentTensor.Pow(scalarExponentF);
            }
            else if (exponent is double scalarExponent)
            {
                result = this.currentTensor.Pow(scalarExponent);
            }
            else if (exponent is Tensor tensorExponent)
            {
                exponentTensor = tensorExponent;
                result = this.currentTensor.Pow(tensorExponent);
            }
            else
            {
                throw new ArgumentException("Exponent must be either a double or a Tensor.");
            }

            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });
            var grad = Tensor.ToTensorArray(exponentTensor != null ? 2 : 1, this.currentTensor.Shape);

            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.PowReverse(upstreamGrad, exponent);
                PradOp?[] ops = new PradOp?[gradients.Length];
                var tensors = new Tensor[] { this.currentTensor };
                if (exponentTensor != null)
                {
                    tensors = tensors.Append(exponentTensor).ToArray();
                }

                for (int i = 0; i < gradients.Length; i++)
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs an element-wise minimum operation between this tensor and the provided tensor.
        /// </summary>
        /// <param name="other">The tensor to compare with the current tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> containing the result tensor of the element-wise minimum operation
        /// and the associated gradients for backpropagation.
        /// </returns>
        /// <remarks>
        /// This method computes the element-wise minimum of two tensors, stores the result in the current tensor,
        /// and registers a backpropagation step to calculate the reverse gradient during the backpropagation phase.
        /// </remarks>
        /// <example>
        /// <code>
        /// var tensor1 = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        /// var tensor2 = new Tensor(new int[] { 2, 2 }, new double[] { 2, 1, 4, 3 });
        /// var result = tensor1.Min(tensor2);
        /// // Result tensor: { 1, 1, 3, 3 }
        /// </code>
        /// </example>
        [PradOperation(nameof(MinOp))]
        public PradResult Min(Tensor other)
        {
            var result = this.currentTensor.Min(other);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.MinReverse(upstreamGrad, other);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, other };
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Creates an "On-Off" embedding using a learned sparsity tensor T.
        /// </summary>
        /// <param name="indices">Tensor of indices, representing which rows to select.</param>
        /// <param name="binaryCondition">Tensor with binary values (0 or 1) indicating the condition for each index.</param>
        /// <param name="sparsityTensor">A learned tensor of shape [1, N] providing sparsity values for embedding interleaving.</param>
        /// <returns>A new tensor with doubled column size, applying the alternating pattern based on binaryCondition and sparsityTensor.</returns>
        /// <exception cref="ArgumentException">Thrown if indices and binaryCondition shapes don't match, or if binaryCondition contains values other than 0 or 1, or if sparsityTensor shape is incompatible.</exception>
        [PradOperation(nameof(OnOffEmbeddingOp))]
        public PradResult OnOffEmbedding(Tensor indices, Tensor binaryCondition, Tensor sparsityTensor)
        {
            var result = this.currentTensor.OnOffEmbedding(indices, binaryCondition, sparsityTensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var (gradient1, gradient2) = tensorReverse.OnOffEmbeddingReverse(upstreamGrad, indices, binaryCondition);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, sparsityTensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (new Tensor[] { gradient1, gradient2 }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Generates all possible pairings between two 1D tensors.
        /// Optimized using Array.Fill, Array.Copy, and Parallel.For for efficient memory operations and parallelism.
        /// </summary>
        /// <param name="other">A tensor of shape [1, P].</param>
        /// <returns>A tensor of shape [2, N * P] where each column represents a pairing between the tensors.</returns>
        [PradOperation(nameof(PairwiseTileOp))]
        public PradResult PairwiseTile(Tensor other)
        {
            var result = this.currentTensor.PairwiseTile(other);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor, other });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var (gradient1, gradient2) = tensorReverse.PairwiseTileReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, other };
                for (int i = 0; i < grad.Length; i++)
                {
                    var tensor = tensors[i];
                    if (tensor is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (new Tensor[] { gradient1, gradient2 }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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

            this.splitGradients = new Tensor[results.Length];
            this.splitStep = split;

            newOp.originalSplitOp = this;
            newOp.splitIndex = 0;
            newOp.splitGradients = this.splitGradients;
            newOp.gradientStackCounter = 0;

            var branches = newOp.SplitBranchFromResults(this, results);
            var splitOps = new PradOp[] { newOp }.Concat(branches).ToArray();

            this.splitOps = splitOps;

            this.gradientStackCounter = 0;
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
            var initialShape = (int[])this.currentTensor.Shape.Clone();
            var result = this.currentTensor.BroadcastTo(newShape);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, initialShape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.BroadcastToReverse(upstreamGrad, initialShape);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            var result = this.currentTensor.Exclude(PradTools.Cast(min), PradTools.Cast(max));
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ExcludeReverse(upstreamGrad, PradTools.Cast(min), PradTools.Cast(max));
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            var result = this.currentTensor.Clip(PradTools.Cast(min), PradTools.Cast(max));
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ClipReverse(upstreamGrad, PradTools.Cast(min), PradTools.Cast(max));
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Gathers slices from the tensor along the specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="skip">The indices to skip.</param>
        /// <param name="restart">The indices where to restart.</param>
        /// <returns>The gathered tensor along with the gradient placeholders.</returns>
        [PradOperation(nameof(InterleavedGatherOp))]
        public PradResult InterleavedGather(int skip, int restart)
        {
            var result = this.currentTensor.InterleavedGather(skip, restart);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.InterleavedGatherReverse(upstreamGrad, skip, restart);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Gathers slices from the tensor along the specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="skip">The indices to skip.</param>
        /// <param name="restart">The indices where to restart.</param>
        /// <returns>The gathered tensor along with the gradient placeholders.</returns>
        [PradOperation(nameof(InterleavedGatherInverseOp))]
        public PradResult InterleavedGatherInverse(int skip, int restart)
        {
            var result = this.currentTensor.InterleavedGatherInverse(skip, restart);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.InterleavedGatherInverseReverse(upstreamGrad, skip, restart);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise "equals" comparison between the current tensor and another tensor.
        /// </summary>
        /// <param name="tensor">The tensor to compare against the current tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> representing the result of the "equals" operation,
        /// along with the associated gradient and backpropagation steps.
        /// </returns>
        /// <remarks>
        /// This operation compares the elements of the current tensor with the corresponding elements
        /// of the provided <paramref name="tensor"/>. The result tensor contains boolean-like values
        /// indicating where the elements of the current tensor are equal to the elements of the
        /// provided tensor. The gradient for this operation is always zero because the "equals"
        /// operation is non-differentiable.
        /// </remarks>
        [PradOperation(nameof(EqualsOp))]
        public PradResult Equals(Tensor tensor)
        {
            var result = this.currentTensor.Equals(tensor);
            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                // The gradient for Equals is always zero, but we need to pass it back
                var gradients = new Tensor[] { new Tensor(this.currentTensor.Shape), new Tensor(tensor.Shape) };
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    if (tensors[i] is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise "less than" comparison between the current tensor and another tensor.
        /// </summary>
        /// <param name="tensor">The tensor to compare against the current tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> representing the result of the "less than" operation,
        /// along with the associated gradient and backpropagation steps.
        /// </returns>
        /// <remarks>
        /// This operation compares the elements of the current tensor with the corresponding elements
        /// of the provided <paramref name="tensor"/>. The result tensor contains boolean-like values
        /// indicating where the elements of the current tensor are less than the elements of the
        /// provided tensor. The gradient for this operation is always zero because the "less than"
        /// operation is non-differentiable.
        /// </remarks>
        [PradOperation(nameof(LessThanOp))]
        public PradResult LessThan(Tensor tensor)
        {
            var result = this.currentTensor.LessThan(tensor);
            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                // The gradient for LessThan is always zero, but we need to pass it back
                var gradients = new Tensor[] { new Tensor(this.currentTensor.Shape), new Tensor(tensor.Shape) };
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    if (tensors[i] is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise "greater than" comparison between the current tensor and another tensor.
        /// </summary>
        /// <param name="tensor">The tensor to compare against the current tensor.</param>
        /// <returns>
        /// A <see cref="PradResult"/> representing the result of the "greater than" operation,
        /// along with the associated gradient and backpropagation steps.
        /// </returns>
        /// <remarks>
        /// This operation compares the elements of the current tensor with the corresponding elements
        /// of the provided <paramref name="tensor"/>. The result tensor contains boolean-like values
        /// indicating where the elements of the current tensor are greater than the elements of the
        /// provided tensor. The gradient for this operation is always zero because the "greater than"
        /// operation is non-differentiable.
        /// </remarks>
        [PradOperation(nameof(GreaterThanOp))]
        public PradResult GreaterThan(Tensor tensor)
        {
            var result = this.currentTensor.GreaterThan(tensor);
            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                // The gradient for GreaterThan is always zero, but we need to pass it back
                var gradients = new Tensor[] { new Tensor(this.currentTensor.Shape), new Tensor(tensor.Shape) };
                PradOp?[] ops = new PradOp?[2];
                var tensors = new Tensor[] { this.currentTensor, tensor };
                for (int i = 0; i < grad.Length; i++)
                {
                    if (tensors[i] is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Selects elements from the current tensor or another tensor based on a condition tensor.
        /// </summary>
        /// <param name="condition">A tensor containing boolean-like values (0 or 1), where 1 indicates
        /// that the corresponding element should be taken from the current tensor, and 0 indicates that
        /// the corresponding element should be taken from the <paramref name="other"/> tensor.</param>
        /// <param name="other">The tensor from which to select elements when the condition is false (0).</param>
        /// <returns>
        /// A <see cref="PradResult"/> representing the result of the "where" operation, along with the
        /// associated gradient and backpropagation steps.
        /// </returns>
        /// <remarks>
        /// This operation creates a new tensor by selecting elements from the current tensor and
        /// the <paramref name="other"/> tensor based on the values in the <paramref name="condition"/> tensor.
        /// The gradient for the "where" operation is computed based on the condition tensor, with the gradient
        /// of the condition itself being zero.
        /// </remarks>
        [PradOperation(nameof(WhereOp))]
        public PradResult Where(Tensor condition, Tensor other)
        {
            var result = Tensor.Where(condition, this.currentTensor, other);
            var grad = Tensor.ToTensorArray(3, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradX = Tensor.Where(condition, upstreamGrad, new Tensor(upstreamGrad.Shape));
                var gradY = Tensor.Where(condition.VectorizedBitFlip(), upstreamGrad, new Tensor(upstreamGrad.Shape));
                var gradCondition = new Tensor(condition.Shape); // Gradient of condition is always zero

                var gradients = new Tensor[] { gradX, gradCondition, gradY };
                PradOp?[] ops = new PradOp?[3];
                var tensors = new Tensor[] { this.currentTensor, condition, other };
                for (int i = 0; i < grad.Length; i++)
                {
                    if (tensors[i] is PradTensor pradTensor)
                    {
                        ops[i] = pradTensor.PradOp;
                    }
                }

                return (gradients, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Performs an element-wise modulus operation between the current tensor and the provided tensor.
        /// </summary>
        /// <param name="tensor">The tensor to perform the modulus operation with.</param>
        /// <returns>
        /// A <see cref="PradResult"/> containing the result of the element-wise modulus operation
        /// and managing the backpropagation steps for gradient calculation.
        /// </returns>
        /// <remarks>
        /// The modulus operation is defined as the remainder of the division of the current tensor by the provided tensor.
        /// This method also sets up the backpropagation logic to calculate gradients during the training of a neural network.
        /// The gradient w.r.t. the first operand (current tensor) is straightforwardly the upstream gradient.
        /// The gradient w.r.t. the second operand (provided tensor) involves a more complex calculation,
        /// which includes an element-wise floor operation followed by negation.
        /// </remarks>
        /// <example>
        /// Suppose you have a tensor `A` with values [4, 7, 9] and tensor `B` with values [2, 3, 5].
        /// The result of `A.Modulus(B)` will be a tensor with values [0, 1, 4], representing the element-wise modulus.
        /// </example>
        /// <exception cref="ArgumentException">Thrown if the shapes of the tensors do not match for the operation.</exception>
        [PradOperation(nameof(ModulusOp))]
        public PradResult Modulus(Tensor tensor)
        {
            var result = this.currentTensor.Modulus(tensor);
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(2, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ModulusReverse(upstreamGrad, tensor);
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            if (tensor is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Computes the element-wise arc cosine (inverse cosine) of the tensor using MKL.NET.
        /// </summary>
        /// <returns>A new tensor with the element-wise arc cosine values.</returns>
        [PradOperation(nameof(ArcCosOp))]
        public PradResult ArcCos()
        {
            var result = this.currentTensor.ElementwiseArcCos();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.ElementwiseArcCosReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Multiplies the columns of the tensor and records the operation for backpropagation.
        /// </summary>
        /// <returns>The tensor with multiplied columns along with the gradient placeholders.</returns>
        [PradOperation(nameof(MultiplyColumnsOp))]
        public PradResult MultiplyColumns()
        {
            var result = this.currentTensor.MultiplyColumns();
            var tensorReverse = new TensorReverse(new Tensor[] { this.currentTensor });

            var grad = Tensor.ToTensorArray(1, this.currentTensor.Shape);
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradient = tensorReverse.MultiplyColumnsReverse(upstreamGrad);
                PradOp?[] ops = new PradOp?[1];
                return (new Tensor[] { gradient }, ops);
            };

            var pradResult = new PradResult(this, result, grad);
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
            this.currentTensor = result;
            return pradResult;
        }

        /// <summary>
        /// Concatenates the current tensor with other tensors along a specified axis and records the operation for backpropagation.
        /// </summary>
        /// <param name="tensors">The tensors to concatenate.</param>
        /// <param name="axis">The axis along which to concatenate.</param>
        /// <param name="reordering">The order of concatenation.</param>
        /// <returns>The concatenated tensor along with the gradient placeholders.</returns>
        [PradOperation(nameof(ConcatOp))]
        public PradResult Concat(Tensor[] tensors, int axis = 0, int[]? reordering = null)
        {
            if (tensors[0] is PradTensor pradTensor)
            {
                pradTensor.PradOp.LinkedBranches.Add(this);
            }

            if (reordering != null && reordering.Length == 0)
            {
                reordering = null;
            }

            var combinedTensors = new Tensor[tensors.Length + 1];
            combinedTensors[0] = this.currentTensor;
            Array.Copy(tensors, 0, combinedTensors, 1, tensors.Length);

            var result = Tensor.Concat(combinedTensors, axis, reordering);
            var tensorReverse = new TensorReverse(combinedTensors);

            var grads = combinedTensors.Select(t => new Tensor(t.Shape)).ToArray();
            Func<Tensor, (Tensor[], PradOp?[])> backpropStep = upstreamGrad =>
            {
                var gradients = tensorReverse.ConcatReverse(upstreamGrad, axis, reordering);
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
            this.BackpropagationSteps.Add((backpropStep, pradResult));
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
                pradOps[i].SetBranchTracker(this.branchTracker);
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

            // Clear last gradient if starting fresh
            if (!this.IsStarted && !this.IsFinished)
            {
                this.lastSeedGradient = null !;
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

            if (this.IsStarted)
            {
                if (this.BackpropagationSteps.Count == 0)
                {
                    this.LastGradient = this.UpstreamGradient;
                    return this.UpstreamGradient;
                }

                Console.WriteLine("Backpropagation has already been started for this branch.");
                return this.UpstreamGradient;
            }

            if (this.IsDependentBranch)
            {
                var attribute = (ConditionallyInternalUseOnlyAttribute)Attribute.GetCustomAttribute(
                    MethodBase.GetCurrentMethod(), typeof(ConditionallyInternalUseOnlyAttribute));

                attribute?.Validate(this.IsDependentBranch, "Back should not be called on a dependent branch.");
            }

            // if (this.splitOps != null)
            // {
            //    var gradientLeft = this.splitOps[0].Back(this.UpstreamGradient);
            //    var gradientRight = this.splitOps[1].Back();

            // if (this.splitStep.HasValue)
            //    {
            //        this.UpstreamGradient = this.splitStep.Value.splitStep(new Tensor[] { gradientLeft, gradientRight });
            //    }
            // }
            Tensor currentUpstream = this.UpstreamGradient;

            this.IsStarted = true;
            this.StartDate = DateTimeOffset.UtcNow;

            // Reverse iterate over backpropagation steps to accumulate gradients
            foreach (var (step, result) in this.BackpropagationSteps.AsEnumerable().Reverse())
            {
                var isDependentBranch = result.PradOp.IsDependentBranch;
                var linkedBranches = result.PradOp.LinkedBranches;

                // First, backpropagate through all branches
                foreach (var branch in result.Branches)
                {
                    if (branch.UpstreamGradient == null)
                    {
                        if (branch.ResetDate > branch.StartDate)
                        {
                            continue;
                        }

                        var branchResult = branch.LinkedBranches.FirstOrDefault();
                        if (branchResult != null)
                        {
                            if (branchResult.UpstreamGradient != null)
                            {
                                branchResult.Back();
                            }
                            else if (branchResult.LinkedBranches.Any())
                            {
                                var linkedBranchResult = branchResult.LinkedBranches.First();
                                if (linkedBranchResult.UpstreamGradient != null)
                                {
                                    linkedBranchResult.Back();
                                }

                                if (branchResult.UpstreamGradient != null)
                                {
                                    branchResult.Back();
                                }
                            }
                        }
                    }

                    if (branch.UpstreamGradient == null)
                    {
                        var lBranches = linkedBranches.Where(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null);
                        foreach (var lBranch in lBranches)
                        {
                            lBranch.Back();
                        }

                        if (branch.UpstreamGradient == null && branch.BackpropagationSteps.Any())
                        {
                            var bstep = branch.BackpropagationSteps.LastOrDefault();
                            (Func<Tensor, (Tensor[], PradOp?[])> backpropStep, PradResult result)? linkedStep = null;
                            if (branch.LinkedBranches.Any())
                            {
                                linkedStep = branch.LinkedBranches.First().BackpropagationSteps.LastOrDefault();
                            }

                            this.branchTracker.ProcessBranch(branch, bstep.result.PradOp, linkedStep == null ? default : linkedStep.GetValueOrDefault().result.PradOp);
                        }

                        if (branch.UpstreamGradient == null)
                        {
                            this.branchTracker.RunBranchesFor(branch);
                        }

                        if (branch.UpstreamGradient == null)
                        {
                            foreach (var lBranch in linkedBranches.Concat(branch.LinkedBranches))
                            {
                                if (!lBranch.IsStarted && !lBranch.IsFinished && lBranch.UpstreamGradient != null)
                                {
                                    lBranch.Back();
                                }

                                if (lBranch.branchTracker != this.branchTracker)
                                {
                                    lBranch.branchTracker.RunBranchesFor(branch);
                                }

                                if (branch.UpstreamGradient == null && lBranch.splitOps != null)
                                {
                                    foreach (var splitOp in lBranch.splitOps)
                                    {
                                        splitOp.branchTracker.RunBranchesFor(branch);
                                    }
                                }
                            }
                        }

                        if (branch.UpstreamGradient == null)
                        {
                            this.branchTracker.RunBranchesFor(branch);
                        }

                        if (branch.UpstreamGradient == null && branch.splitOps != null)
                        {
                            foreach (var splitOp in branch.splitOps)
                            {
                                splitOp.branchTracker.RunBranchesFor(branch);
                            }
                        }

                        if (branch.UpstreamGradient == null)
                        {
                            if (branch.BackpropagationSteps.Any())
                            {
                                throw new InvalidOperationException($"An error occurred during backpropagation. Branch '{branch.Id}' has no upstream gradient.");
                            }
                            else
                            {
                                branch.IsStarted = true;
                                branch.IsFinished = true;
                                branch.StartDate = DateTimeOffset.UtcNow;
                                branch.FinishDate = DateTimeOffset.UtcNow;
                            }
                        }
                    }

                    if (branch.IsFinished || branch.ResetDate > branch.StartDate)
                    {
                        currentUpstream = currentUpstream.ElementwiseAdd(branch.LastGradient);
                    }
                    else
                    {
                        var branchGradient = branch.Back();
                        if (branch.IsFinished || branch.ResetDate > branch.StartDate)
                        {
                            currentUpstream = currentUpstream.ElementwiseAdd(branchGradient);
                        }
                        else
                        {
                            branch.WaitingToAdd.Enqueue(currentUpstream);
                        }
                    }
                }

                // Then, backpropagate through all split branches
                List<Tensor> branchGradients = new List<Tensor>();
                foreach (var branch in result.SplitBranches.Where(x => x.UpstreamGradient != null))
                {
                    if (branch.IsFinished || branch.ResetDate > branch.StartDate)
                    {
                        branchGradients.Add(branch.LastGradient);
                    }
                    else
                    {
                        var branchGradient = branch.Back();
                        branchGradients.Add(branchGradient);
                    }
                }

                var (gradients, ops) = step(currentUpstream);
                GradientRecorder.Instance.RecordGradient(step.Method.Name, gradients);

                foreach (var branchGradient in branchGradients)
                {
                    gradients[0] = gradients[0].ElementwiseAdd(branchGradient);
                }

                currentUpstream = gradients[0];

                Parallel.For(0, result.Gradients.Length, i =>
                {
                    var s = step;
                    result.Gradients[i] = result.Gradients[i].ElementwiseAdd(gradients[i]);
                    if (ops[i] != null)
                    {
                        if (ops[i]!.IsStarted && !ops[i]!.IsFinished)
                        {
                            throw new ArgumentException("Did you forget to 'Branch'? PradOp ID:" + ops[i]!.Id.ToString());
                        }

                        ops[i]?.SetUpstreamGradient(gradients[i]);
                        if (ops[i]!.IsHighPriorityForBackpropagation || (!ops[i]!.IsDependentBranch && !ops[i]!.IsLowPriorityForBackpropagation))
                        {
                            ops[i]?.Back();
                        }
                        else if (!this.branchTracker.VisitedBranches.Contains(ops[i]!))
                        {
                            this.branchTracker.VisitedBranches.Add(ops[i]!);
                        }
                    }
                });
            }

            foreach (var branch in this.initialResult.Branches)
            {
                var branchGradient = branch.Back();
                currentUpstream = currentUpstream.ElementwiseAdd(branchGradient);
            }

            PradOp opWithSplit = this.originalSplitOp ?? this;

            if (opWithSplit.splitOps != null && opWithSplit.splitStep.HasValue)
            {
                // Save the gradient at the correct index
                opWithSplit.splitGradients[this.splitIndex] = currentUpstream;

                // Increment the stack counter on the original PradOp
                opWithSplit.gradientStackCounter++;

                // Check if all gradients have been collected
                if (opWithSplit.gradientStackCounter == opWithSplit.splitGradients.Length)
                {
                    // Combine gradients using the splitStep function
                    Tensor combinedGradient = opWithSplit.splitStep.Value.splitStep(opWithSplit.splitGradients);
                    GradientRecorder.Instance.RecordGradient(opWithSplit.splitStep.Value.splitStep.Method.Name, new Tensor[] { combinedGradient });

                    // Reset the stack counter for potential reuse
                    opWithSplit.gradientStackCounter = 0;

                    // Start backpropagation on the main PradOp instance
                    opWithSplit.UpstreamGradient = combinedGradient;
                    var splitOps = opWithSplit.splitOps;
                    opWithSplit.splitOps = null;
                    opWithSplit.Back();
                    opWithSplit.splitOps = splitOps;
                }
            }

            if (this.BackpropagationMode == BackpropagationMode.Accumulate)
            {
                this.seedGradientHistory.Add(currentUpstream);
                this.LastGradient = this.LastGradient.ElementwiseAdd(currentUpstream);
            }
            else
            {
                this.LastGradient = currentUpstream;
            }

            this.IsFinished = true;
            this.FinishDate = DateTimeOffset.UtcNow;

            if (this.WaitingToAdd.Count > 0)
            {
                for (int w = 0; w < this.WaitingToAdd.Count; ++w)
                {
                    var waiting = this.WaitingToAdd.Dequeue();
                    waiting.ElementwiseAdd(currentUpstream);
                }
            }

            this.ResetBackpropagation();

            return currentUpstream;
        }

        /// <summary>
        /// Clips the gradients.
        /// </summary>
        /// <param name="clipper">The gradient clipper.</param>
        public void ClipGradients(IClipper clipper)
        {
            clipper.ClipGradients(this.SeedGradient);
        }

        /// <summary>
        /// Optimize the weights with an optimizer.
        /// </summary>
        /// <param name="optimizer">The optimizer to use.</param>
        public void Optimize(IOptimizer optimizer)
        {
            if (!this.IsFinished && this.ResetDate < this.StartDate)
            {
                throw new InvalidOperationException("This branch has not finished calculating 'Back'.");
            }

            if (!optimizer.IsInitialized)
            {
                optimizer.Initialize(this.seed);
            }

            optimizer.UpdateWeights(this.seed, this.SeedGradient);
        }

        /// <summary>
        /// Is the result currently associated with the PradOp instance.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <returns>A value indicating whether the PradOp instance for the result has not been modified.</returns>
        internal bool IsCurrentlyAssociated(PradResult result)
        {
            return this.BackpropagationSteps.Any() && this.BackpropagationSteps.Last().result == result;
        }

        /// <summary>
        /// Branch to another prad op after the fact.
        /// </summary>
        /// <param name="parentResult">The parent result.</param>
        /// <returns>The other prad op.</returns>
        internal PradOp BranchAfterTheFact(PradResult parentResult)
        {
            var branchedOp = new PradOp(parentResult.ResultTensor);
            if (this.BackpropagationSteps.Any())
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
            if (this.BackpropagationSteps.Any())
            {
                branchedOp.parentResult = this.BackpropagationSteps.Last().result;
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
        /// <param name="originalPradOp">The original prad op.</param>
        /// <param name="results">The results.</param>
        /// <returns>The other prad op.</returns>
        internal PradOp[] SplitBranchFromResults(PradOp originalPradOp, Tensor[] results)
        {
            List<PradOp> splits = new List<PradOp>();
            for (int i = 1; i < results.Length; ++i)
            {
                var branchedOp = new PradOp(results[i]);
                branchedOp.originalSplitOp = originalPradOp;
                branchedOp.splitIndex = i;
                branchedOp.splitGradients = originalPradOp.splitGradients;
                branchedOp.gradientStackCounter = 0;
                branchedOp.LinkedBranches.Add(originalPradOp);

                if (this.BackpropagationSteps.Any())
                {
                    branchedOp.parentResult = this.BackpropagationSteps.Last().result;
                    branchedOp.parentResult.SplitBranches.Add(branchedOp);
                }
                else if (this.splitStep.HasValue)
                {
                    branchedOp.parentResult = this.splitStep.Value.result;
                }

                splits.Add(branchedOp);
            }

            return splits.ToArray();
        }

        /// <summary>
        /// Branch to another prad op.
        /// </summary>
        /// <param name="originalOp">The original prad op.</param>
        /// <returns>The other prad op.</returns>
        internal PradOp RecordSplitBranch(PradOp originalOp)
        {
            if (this.BackpropagationSteps.Any())
            {
                var parentResult = originalOp.BackpropagationSteps.Last().result;
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

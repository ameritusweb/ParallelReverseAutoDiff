//------------------------------------------------------------------------------
// <copyright file="OperationBase.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;
    using System.Xml.Linq;

    /// <inheritdoc />
    public abstract class OperationBase : IOperationBase
    {
        private ConcurrentDictionary<Guid, Matrix> intermediateMatrices;
        private ConcurrentDictionary<Guid, Matrix[]> intermediateMatrixArrays;
        private ConcurrentDictionary<Guid, DeepMatrix> intermediateDeepMatrices;
        private ConcurrentDictionary<Guid, DeepMatrix[]> intermediateDeepMatrixArrays;
        private ConcurrentDictionary<Guid, object[]> intermediateObjectArrays;

        /// <summary>
        /// Initializes a new instance of the <see cref="OperationBase"/> class.
        /// </summary>
        protected OperationBase()
        {
            // Initialize properties
            this.Inputs = new List<string>();
            this.Outputs = new List<string>();
            this.BackwardAdjacentOperations = new List<IOperationBase?>();
            this.BackwardDependencyCounts = new List<int>();
            this.AccumulatedGradients = new List<BackwardResult>();
            this.Tasks = new List<Task>();
            this.BackwardDependencies = new List<List<string>>();
            this.VisitedFrom = new List<string>();
        }

        /// <inheritdoc />
        public bool IsComplete { get; set; }

        /// <inheritdoc />
        public int TimeStepIndex { get; set; } = -1;

        /// <inheritdoc />
        public int LayerIndex { get; set; } = -1;

        /// <inheritdoc />
        public Type OperationType { get; set; }

        /// <inheritdoc />
        public bool HasNext
        {
            get
            {
                return this.Next != null;
            }
        }

        /// <inheritdoc />
        public IOperationBase Next { get; set; }

        /// <inheritdoc />
        public string Id { get; set; }

        /// <inheritdoc />
        public string SpecificId { get; set; }

        /// <inheritdoc />
        public object[] GradientDestinations { get; set; }

        /// <inheritdoc />
        public string ResultToName { get; set; }

        /// <inheritdoc />
        public object[] Parameters { get; set; }

        /// <inheritdoc />
        public List<Task> Tasks { get; set; }

        /// <inheritdoc />
        public List<string> Inputs { get; set; }

        /// <inheritdoc />
        public List<string> Outputs { get; set; }

        /// <inheritdoc />
        public object BackwardInput { get; set; }

        /// <inheritdoc />
        public List<List<string>> BackwardDependencies { get; set; }

        /// <inheritdoc />
        public List<string> VisitedFrom { get; set; }

        /// <inheritdoc />
        public List<IOperationBase?> BackwardAdjacentOperations { get; set; }

        /// <inheritdoc />
        public List<int> BackwardDependencyCounts { get; set; }

        /// <inheritdoc />
        public List<BackwardResult> AccumulatedGradients { get; set; }

        /// <inheritdoc />
        public object?[] CalculatedGradient { get; set; }

        /// <inheritdoc />
        public int OutputDependencyCount { get; set; }

        /// <inheritdoc />
        public int VisitedCount { get; set; }

        /// <inheritdoc />
        public ReaderWriterLockSlim Lock { get; set; }

        /// <inheritdoc />
        public SemaphoreSlim? SyncSemaphore { get; set; }

        /// <inheritdoc />
        public LayerInfo LayerInfo { get; set; }

        /// <summary>
        /// Gets or sets the property to store the output of the operation.
        /// </summary>
        protected Matrix Output { get; set; }

        /// <summary>
        /// Gets or sets the property to store the deep output of the operation.
        /// </summary>
        protected DeepMatrix DeepOutput { get; set; }

        /// <summary>
        /// Gets the property to store the intermediate matrices of the operation.
        /// </summary>
        protected ConcurrentDictionary<Guid, Matrix> IntermediateMatrices
        {
            get
            {
                return this.intermediateMatrices ??= new ConcurrentDictionary<Guid, Matrix>();
            }
        }

        /// <summary>
        /// Gets the property to store the intermediate deep matrices of the operation.
        /// </summary>
        protected ConcurrentDictionary<Guid, DeepMatrix> IntermediateDeepMatrices
        {
            get
            {
                return this.intermediateDeepMatrices ??= new ConcurrentDictionary<Guid, DeepMatrix>();
            }
        }

        /// <summary>
        /// Gets the property to store the intermediate matrix arrays of the operation.
        /// </summary>
        protected ConcurrentDictionary<Guid, Matrix[]> IntermediateMatrixArrays
        {
            get
            {
                return this.intermediateMatrixArrays ??= new ConcurrentDictionary<Guid, Matrix[]>();
            }
        }

        /// <summary>
        /// Gets the property to store the intermediate deep matrix arrays of the operation.
        /// </summary>
        protected ConcurrentDictionary<Guid, DeepMatrix[]> IntermediateDeepMatrixArrays
        {
            get
            {
                return this.intermediateDeepMatrixArrays ??= new ConcurrentDictionary<Guid, DeepMatrix[]>();
            }
        }

        /// <summary>
        /// Gets the property to store the intermediate object arrays of the operation.
        /// </summary>
        protected ConcurrentDictionary<Guid, object[]> IntermediateObjectArrays
        {
            get
            {
                return this.intermediateObjectArrays ??= new ConcurrentDictionary<Guid, object[]>();
            }
        }

        /// <inheritdoc />
        public void InitializeFrom(OperationInfo info, ConcurrentDictionary<string, Func<LayerInfo, object>> gradients, LayerInfo layerInfo)
        {
            this.Id = info.Id;
            this.SpecificId = info.Id + layerInfo.ToString();
            this.LayerInfo = layerInfo;
            this.OperationType = this.GetType();
            this.Inputs = info.Inputs.ToList();
            string resultTo = info.SetResultTo;
            if (resultTo != null)
            {
                this.ResultToName = resultTo;
            }

            string[] gradientResultTo = info.GradientResultTo;
            if (gradientResultTo != null)
            {
                this.GradientDestinations = new object[gradientResultTo.Length];
                for (int i = 0; i < gradientResultTo.Length; ++i)
                {
                    if (gradientResultTo[i] != null)
                    {
                        string[] split = gradientResultTo[i].Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                        this.GradientDestinations[i] = gradients[split[0]](layerInfo);
                    }
                }
            }
        }

        /// <inheritdoc />
        public void Reset()
        {
            this.VisitedCount = 0;
            this.VisitedFrom?.Clear();
            this.AccumulatedGradients.Clear();
            this.SyncSemaphore?.Dispose();
            this.SyncSemaphore = null;
            this.IsComplete = false;
            this.Tasks?.Clear();
        }

        /// <inheritdoc />
        public virtual void Store(Guid id)
        {
        }

        /// <inheritdoc />
        public virtual void Restore(Guid id)
        {
        }

        /// <inheritdoc />
        public virtual Matrix GetOutput()
        {
            return this.Output;
        }

        /// <inheritdoc />
        public virtual DeepMatrix GetDeepOutput()
        {
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public virtual void AccumulateGradient(object?[] gradients)
        {
            if (this.GradientDestinations != null && this.GradientDestinations.Length > 0)
            {
                for (int dest = 0; dest < this.GradientDestinations.Length; ++dest)
                {
                    var gradientResultTo = this.GradientDestinations[dest];
                    if (gradientResultTo != null)
                    {
                        if (gradientResultTo is Matrix matrixGradientResult)
                        {
                            if (dest >= gradients.Length)
                            {
                                throw new InvalidOperationException("The output gradient must be non-null.");
                            }

                            var output = (Matrix?)gradients[dest];
                            if (output == null)
                            {
                                throw new InvalidOperationException("The output gradient must be non-null.");
                            }

                            int numRows = matrixGradientResult.Rows;
                            int numCols = matrixGradientResult.Cols;
                            for (int i = 0; i < numRows; ++i)
                            {
                                for (int j = 0; j < numCols; ++j)
                                {
                                    matrixGradientResult[i][j] += output[i][j];
                                }
                            }
                        }
                        else if (gradientResultTo is DeepMatrix deepMatrixGradientResult)
                        {
                            var output = (DeepMatrix?)gradients[dest];
                            if (output == null)
                            {
                                throw new InvalidOperationException("The output gradient must be non-null.");
                            }

                            int depth = deepMatrixGradientResult.Depth;
                            int numRows = deepMatrixGradientResult.Rows;
                            int numCols = deepMatrixGradientResult.Cols;
                            for (int d = 0; d < depth; ++d)
                            {
                                for (int i = 0; i < numRows; ++i)
                                {
                                    for (int j = 0; j < numCols; ++j)
                                    {
                                        deepMatrixGradientResult[d][i][j] += output[d][i][j];
                                    }
                                }
                            }
                        }
                        else if (gradientResultTo is DeepMatrix[] deepMatrixArrayGradientResult)
                        {
                            var output = (DeepMatrix[]?)gradients[dest];
                            if (output == null)
                            {
                                throw new InvalidOperationException("The output gradient must be non-null.");
                            }

                            int length = deepMatrixArrayGradientResult.Length;
                            int depth = deepMatrixArrayGradientResult[0].Depth;
                            int numRows = deepMatrixArrayGradientResult[0].Rows;
                            int numCols = deepMatrixArrayGradientResult[0].Cols;
                            for (int l = 0; l < length; ++l)
                            {
                                for (int d = 0; d < depth; ++d)
                                {
                                    for (int i = 0; i < numRows; ++i)
                                    {
                                        for (int j = 0; j < numCols; ++j)
                                        {
                                            deepMatrixArrayGradientResult[l][d][i][j] += output[l][d][i][j];
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            throw new InvalidOperationException("The output gradient must be a matrix or a deep matrix.");
                        }
                    }
                }
            }
        }

        /// <inheritdoc />
        public virtual void ResultTo(Func<int, object> func)
        {
            var oo = func(this.LayerIndex);
            this.CopyResult(oo);
        }

        /// <inheritdoc />
        public virtual void ResultTo(Func<int, int, object> func)
        {
            var oo = func(this.TimeStepIndex, this.LayerIndex);
            this.CopyResult(oo);
        }

        /// <inheritdoc />
        public virtual void ResultTo(ComputationGraph graph)
        {
            var split = this.ResultToName.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
            var oo = graph[MatrixType.Intermediate, split[0], this.LayerInfo];
            this.CopyResult(oo);
        }

        /// <inheritdoc />
        public virtual void Initialize(int startingPointIndex)
        {
            this.OutputDependencyCount = this.BackwardDependencyCounts[startingPointIndex];
            this.InitializeLock();
            this.InitializeSyncSemaphore();
        }

        /// <inheritdoc />
        public virtual void InitializeLock()
        {
            if (this.Lock == null)
            {
                this.Lock = new ReaderWriterLockSlim();
            }
        }

        /// <inheritdoc />
        public virtual void InitializeSyncSemaphore()
        {
            if (this.SyncSemaphore == null)
            {
                this.SyncSemaphore = new SemaphoreSlim(0, this.OutputDependencyCount);
            }
        }

        /// <inheritdoc />
        public void CopyResult(object destination)
        {
            Matrix o;
            if (destination is Operation op)
            {
                o = op.GetOutput();
            }
            else
            {
                o = (Matrix)destination;
            }

            int numRows = this.Output.Length;
            int numCols = this.Output[0].Length;
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    o[i][j] = this.Output[i][j];
                }
            }
        }
    }
}

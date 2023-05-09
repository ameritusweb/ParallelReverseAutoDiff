//------------------------------------------------------------------------------
// <copyright file="Operation.cs" author="ameritusweb" date="5/2/2023">
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

    /// <inheritdoc />
    public abstract class Operation : IOperation
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Operation"/> class.
        /// </summary>
        protected Operation()
        {
            // Initialize properties
            this.Inputs = new List<string>();
            this.Outputs = new List<string>();
            this.BackwardAdjacentOperations = new List<IOperation?>();
            this.BackwardDependencyCounts = new List<int>();
            this.AccumulatedGradients = new List<(Matrix?, Matrix?)>();
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
        public IOperation Next { get; set; }

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
        public List<IOperation?> BackwardAdjacentOperations { get; set; }

        /// <inheritdoc />
        public List<int> BackwardDependencyCounts { get; set; }

        /// <inheritdoc />
        public List<(Matrix?, Matrix?)> AccumulatedGradients { get; set; }

        /// <inheritdoc />
        public (Matrix?, Matrix?) CalculatedGradient { get; set; }

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

        /// <inheritdoc />
        public void InitializeFrom(OperationInfo info, ConcurrentDictionary<string, Func<LayerInfo, Matrix>> gradients, LayerInfo layerInfo)
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
                        this.GradientDestinations[i] = gradients[gradientResultTo[i]](layerInfo);
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
        public virtual Matrix GetOutput()
        {
            return this.Output;
        }

        /// <inheritdoc />
        public abstract (Matrix?, Matrix?) Backward(Matrix dOutput);

        /// <inheritdoc />
        public virtual void AccumulateGradient((Matrix?, Matrix?) dOutput)
        {
            var array3D = MatrixUtils.Reassemble(dOutput).ToList();
            for (int k = array3D.Count; k < this.GradientDestinations.Length; ++k)
            {
                array3D.Add(dOutput.Item2);
            }

            if (this.GradientDestinations != null && this.GradientDestinations.Length > 0)
            {
                for (int d = 0; d < this.GradientDestinations.Length; ++d)
                {
                    var gradientResultTo = (Matrix)this.GradientDestinations[d];
                    if (gradientResultTo != null)
                    {
                        var output = array3D[d];
                        if (output == null)
                        {
                            throw new InvalidOperationException("The output gradient must be non-null.");
                        }

                        int numRows = gradientResultTo.Length;
                        int numCols = gradientResultTo[0].Length;
                        for (int i = 0; i < numRows; ++i)
                        {
                            for (int j = 0; j < numCols; ++j)
                            {
                                gradientResultTo[i][j] += output[i][j];
                            }
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

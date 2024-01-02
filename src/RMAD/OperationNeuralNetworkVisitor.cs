//------------------------------------------------------------------------------
// <copyright file="OperationNeuralNetworkVisitor.cs" author="ameritusweb" date="5/2/2023">
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

    /// <summary>
    /// The operation neural network visitor. Implements the visitor pattern for the backward pass.
    /// </summary>
    public class OperationNeuralNetworkVisitor
    {
        private readonly string id;
        private readonly IOperationBase startNode;
        private readonly int startingPointIndex;
        private readonly List<IOperationBase> operations;
        private readonly ConcurrentDictionary<IOperationBase, (int, bool)> waiting;
        private readonly ConcurrentQueue<Exception> exceptions;
        private bool runInParallel = true;
        private double timeoutInMinutes = 5d;

        /// <summary>
        /// Initializes a new instance of the <see cref="OperationNeuralNetworkVisitor"/> class.
        /// </summary>
        /// <param name="id">An ID to uniquely identify the visitor.</param>
        /// <param name="startNode">The start node for the traveral.</param>
        /// <param name="startingPointIndex">The starting point index.</param>
        public OperationNeuralNetworkVisitor(string id, IOperationBase startNode, int startingPointIndex)
        {
            if (string.IsNullOrEmpty(id))
            {
                throw new ArgumentNullException(nameof(id), $"The parameter {nameof(id)} cannot be null or empty.");
            }

            if (startNode == null)
            {
                throw new ArgumentNullException(nameof(startNode), $"The parameter {nameof(startNode)} cannot be null.");
            }

            if (startingPointIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(startingPointIndex), $"The parameter {nameof(startingPointIndex)} cannot be less than zero.");
            }

            this.id = id;
            this.startNode = startNode;
            this.startingPointIndex = startingPointIndex;
            this.operations = new List<IOperationBase>();
            this.waiting = new ConcurrentDictionary<IOperationBase, (int, bool)>();
            this.exceptions = new ConcurrentQueue<Exception>();
        }

        /// <summary>
        /// Gets or sets a value indicating whether the visitor should run sequentially instead of in parallel.
        /// </summary>
        public bool RunSequentially
        {
            get
            {
                return !this.runInParallel;
            }

            set
            {
                this.runInParallel = !value;
            }
        }

        /// <summary>
        /// Gets the exceptions that occurred during the traversal.
        /// </summary>
        public AggregateException? AggregateException
        {
            get
            {
                if (this.exceptions.Count == 0)
                {
                    return default;
                }

                return new AggregateException(this.exceptions);
            }
        }

        /// <summary>
        /// Gets or sets the timeout in minutes.
        /// </summary>
        public double TimeoutInMinutes
        {
            get
            {
                return this.timeoutInMinutes;
            }

            set
            {
                this.timeoutInMinutes = value;
            }
        }

        /// <summary>
        /// Gets the ID of the visitor.
        /// </summary>
        public string Id
        {
            get
            {
                return this.id;
            }
        }

        /// <summary>
        /// Start the traversal of the backward pass.
        /// </summary>
        /// <returns>The task.</returns>
        public Task TraverseAsync()
        {
            return this.Traverse(this.startNode);
        }

        /// <summary>
        /// Reset the state of the operations.
        /// </summary>
        public void Reset()
        {
            Parallel.For(0, this.operations.Count, i =>
            {
                if (this.operations[i] != null)
                {
                    this.operations[i].Reset();
                }
            });
            this.operations.Clear();
        }

        private async Task Traverse(IOperationBase node, IOperationBase? fromNode = null)
        {
            if (node == null)
            {
                return;
            }

            if (fromNode != null)
            {
                this.operations.Add(node);

                if (node.VisitedFrom.Contains(fromNode.NestedSpecificId))
                {
                    throw new InvalidOperationException("Node must not be visited twice.");
                }

                node.VisitedFrom.Add(fromNode.NestedSpecificId);
            }

            BackwardResult? backwardResult = null;
            BackwardResult[]? initialBackwardResults = null;
            bool pullGradientsDirectly = false;
            if (node is IDeepOperation)
            {
                backwardResult = (node as IDeepOperation)?.Backward((DeepMatrix)node.BackwardInput);
            }
            else if (node is IBatchOperation)
            {
                initialBackwardResults = (node as IBatchOperation)?.Backward((DeepMatrix)node.BackwardInput);
                if (initialBackwardResults == null)
                {
                    throw new InvalidOperationException("Backward results must not be null.");
                }

                if (node is BatchMatrixConcatenateOperation)
                {
                    var param = node.Parameters[0];
                    if (param is IMatrix matrix)
                    {
                        if (matrix.Count == initialBackwardResults.Length)
                        {
                            pullGradientsDirectly = true;
                        }
                    }
                    else if (param is Array array)
                    {
                        if (array.Length == initialBackwardResults.Length)
                        {
                            pullGradientsDirectly = true;
                        }
                    }
                }

                backwardResult = this.CombineResults(initialBackwardResults, node.GradientDestinations, pullGradientsDirectly, node.SwitchFirstTwoDimensions);
            }
            else if (node is IOperation)
            {
                if (node.BackwardInput is DeepMatrix backwardDeep)
                {
                    if (node.LayerInfo.Type == LayerInfoType.Nested)
                    {
                        backwardResult = (node as IOperation)?.Backward(backwardDeep[node.LayerInfo.NestedLayer]);
                    }
                    else
                    {
                        backwardResult = (node as IOperation)?.Backward(backwardDeep[node.LayerInfo.Layer]);
                    }
                }
                else
                {
                    backwardResult = (node as IOperation)?.Backward((Matrix)node.BackwardInput);
                }
            }

            if (backwardResult == null)
            {
                throw new InvalidOperationException("Backward result must not be null.");
            }

            var results = backwardResult.Results;

            bool shouldContinue = false;
            node.Initialize(this.startingPointIndex);
            if (node.SyncSemaphore == null)
            {
                throw new InvalidOperationException("SyncSemaphore must not be null.");
            }

            node.Lock.EnterWriteLock();

            node.VisitedCount++;

            if (node.OutputDependencyCount > 1)
            {
                // Add dOutput to AccumulatedGradients
                node.AccumulatedGradients.Add(backwardResult);
            }

            if (node.VisitedCount == node.OutputDependencyCount)
            {
                shouldContinue = true;
                if (node.OutputDependencyCount > 1)
                {
                    try
                    {
                        this.ReleaseWaiting(node);
                        node.SyncSemaphore.Release(node.OutputDependencyCount - 1);
                    }
                    catch (SemaphoreFullException sfe)
                    {
                        Console.WriteLine("Semaphore full exception");
                        this.exceptions.Enqueue(sfe);
                    }
                }
            }

            node.Lock.ExitWriteLock();

            if (!shouldContinue)
            {
                this.IncrementWaiting(node);
                var semaphoreResult = await node.SyncSemaphore.WaitAsync(TimeSpan.FromMinutes(this.TimeoutInMinutes));
                if (!semaphoreResult)
                {
                    var incrementCount = this.waiting[node].Item1;
                    var isReleased = this.waiting[node].Item2;
                    this.exceptions.Enqueue(new InvalidOperationException($"Semaphore should not exceed timeout. ({incrementCount}, {isReleased}) TimeoutInMinutes is a public property of the visitor that can be changed as needed."));
                }

                return;
            }

            // Perform gradient accumulation here
            if (node.OutputDependencyCount > 1)
            {
                if (node.AccumulatedGradients.Count != node.OutputDependencyCount)
                {
                    throw new InvalidOperationException("Accumulated gradients count must equal the output dependency count.");
                }

                // Accumulate gradients in AccumulatedGradients list
                results = GradientUtils.AccumulateBackwardGradients(node.AccumulatedGradients);
            }

            if (node.GradientDestinations != null)
            {
                node.AccumulateGradient(results, pullGradientsDirectly);
            }

            node.CalculatedGradient = results;

            var adjacentTasks = new List<Task>();
            if (backwardResult.HasMultipleInputs)
            {
                for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
                {
                    IOperationBase? adjacentOperation = node.BackwardAdjacentOperations[i];
                    if (adjacentOperation != null)
                    {
                        adjacentOperation.BackwardInput = results[i] ?? throw new InvalidOperationException("The upstream gradient must not be null.");
                        if (this.runInParallel)
                        {
                            adjacentTasks.Add(Task.Run(() => this.Traverse(adjacentOperation, node)));
                        }
                        else
                        {
                            adjacentTasks.Add(this.Traverse(adjacentOperation, node));
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
                {
                    IOperationBase? adjacentOperation = node.BackwardAdjacentOperations[i];
                    if (adjacentOperation != null)
                    {
                        adjacentOperation.BackwardInput = results[0] ?? throw new InvalidOperationException("The upstream gradient must not be null.");
                        adjacentTasks.Add(this.Traverse(adjacentOperation, node));
                    }
                }
            }

            node.Tasks.AddRange(adjacentTasks);

            await Task.WhenAll(adjacentTasks).ConfigureAwait(false);

            if (!this.operations.Contains(node))
            {
                this.operations.Add(node);
            }

            node.IsComplete = true;
        }

        private BackwardResult CombineResults(BackwardResult[] backwardResults, object[] gradientDestinations, bool pullGradientsDirectly, bool switchFirstTwoDimensions)
        {
            List<object> combinedResults = new List<object>();
            var firstResult = backwardResults[0];
            var hasMultipleInputs = firstResult.HasMultipleInputs;
            var resultsLength = firstResult.Results.Length;
            if (gradientDestinations == null)
            {
                gradientDestinations = new object[resultsLength];
            }

            if (pullGradientsDirectly)
            {
                if (switchFirstTwoDimensions)
                {
                    if (firstResult.Item1 is Matrix)
                    {
                        for (int j = 0; j < resultsLength; ++j)
                        {
                            List<Matrix> matrices = new List<Matrix>();
                            DeepMatrix deepMatrix = new DeepMatrix(backwardResults.Select(x => x.Results[j]).OfType<Matrix>().ToArray());
                            combinedResults.Add(deepMatrix);
                        }

                        return new BackwardResult()
                        {
                            Results = combinedResults.OfType<DeepMatrix>().ToArray(),
                            HasMultipleInputs = hasMultipleInputs,
                        };
                    }
                    else if (firstResult.Item1 is DeepMatrix)
                    {
                        for (int j = 0; j < resultsLength; ++j)
                        {
                            FourDimensionalMatrix deepMatrix = new FourDimensionalMatrix(backwardResults.Select(x => x.Results[j]).OfType<DeepMatrix>().ToArray());
                            combinedResults.Add(deepMatrix);
                        }

                        return new BackwardResult()
                        {
                            Results = combinedResults.OfType<FourDimensionalMatrix>().ToArray(),
                            HasMultipleInputs = hasMultipleInputs,
                        };
                    }
                }
                else
                {
                    if (firstResult.Item1 is Matrix)
                    {
                        for (int j = 0; j < backwardResults.Length; ++j)
                        {
                            var result = new DeepMatrix(backwardResults[j].Results.OfType<Matrix>().ToArray());
                            combinedResults.Add(result);
                        }

                        return new BackwardResult()
                        {
                            Results = combinedResults.OfType<DeepMatrix>().ToArray(),
                            HasMultipleInputs = hasMultipleInputs,
                        };
                    }
                    else if (firstResult.Item1 is DeepMatrix)
                    {
                        for (int j = 0; j < backwardResults.Length; ++j)
                        {
                            var result = new FourDimensionalMatrix(backwardResults[j].Results.OfType<DeepMatrix>().ToArray());
                            combinedResults.Add(result);
                        }

                        return new BackwardResult()
                        {
                            Results = combinedResults.OfType<FourDimensionalMatrix>().ToArray(),
                            HasMultipleInputs = hasMultipleInputs,
                        };
                    }
                }
            }

            var resultTypes = firstResult.Results.Select(r => r?.GetType()).ToArray();
            for (int i = 0; i < resultsLength; ++i)
            {
                var resultType = resultTypes[i];
                var destination = gradientDestinations[i];
                if (destination == null)
                {
                    List<object?> list = new List<object?>();
                    for (int j = 0; j < backwardResults.Length; ++j)
                    {
                        if (backwardResults[j].Results.Length > i)
                        {
                            var result = backwardResults[j].Results[i];
                            list.Add(result);
                        }
                        else
                        {
                            list.Add(null);
                        }
                    }

                    if (resultType == typeof(Matrix))
                    {
                        combinedResults.Add(new DeepMatrix(list.OfType<Matrix>().ToArray()));
                    }
                    else if (resultType == typeof(DeepMatrix))
                    {
                        combinedResults.Add(new FourDimensionalMatrix(list.OfType<DeepMatrix>().ToArray()));
                    }
                }
                else
                {
                    if (resultType == typeof(Matrix))
                    {
                        List<Matrix> averageList = new List<Matrix>();
                        for (int j = 0; j < backwardResults.Length; ++j)
                        {
                            var result = backwardResults[j].Results[i];
                            if (result is Matrix matrix)
                            {
                                averageList.Add(matrix);
                            }
                        }

                        var averaged = averageList.Aggregate((a, b) => a + b) * (1d / averageList.Count);
                        combinedResults.Add(averaged);
                    }
                    else if (resultType == typeof(DeepMatrix))
                    {
                        List<DeepMatrix> averageList = new List<DeepMatrix>();
                        for (int j = 0; j < backwardResults.Length; ++j)
                        {
                            var result = backwardResults[j].Results[i];
                            if (result is DeepMatrix matrix)
                            {
                                averageList.Add(matrix);
                            }
                        }

                        var averaged = averageList.Aggregate((a, b) => a + b) * (1d / averageList.Count);
                        combinedResults.Add(averaged);
                    }
                }
            }

            return new BackwardResult()
            {
                Results = combinedResults.ToArray(),
                HasMultipleInputs = hasMultipleInputs,
            };
        }

        private void IncrementWaiting(IOperationBase operationBase)
        {
            try
            {
                this.waiting.AddOrUpdate(operationBase, (1, false), (k, v) => v.Item2 ? throw new InvalidOperationException("Increment should not be called on released semaphore.") : (v.Item1 + 1, v.Item2));
            }
            catch (InvalidOperationException ioe)
            {
                this.exceptions.Enqueue(ioe);
            }
        }

        private void ReleaseWaiting(IOperationBase operationBase)
        {
            this.waiting.AddOrUpdate(operationBase, (0, true), (k, v) => (v.Item1, true));
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="OperationNeuralNetworkVisitor.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// The operation neural network visitor. Implements the visitor pattern for the backward pass.
    /// </summary>
    public class OperationNeuralNetworkVisitor
    {
        private readonly string id;
        private readonly IOperation startNode;
        private readonly int startingPointIndex;
        private readonly List<IOperation> operations;
        private bool runInParallel = true;

        /// <summary>
        /// Initializes a new instance of the <see cref="OperationNeuralNetworkVisitor"/> class.
        /// </summary>
        /// <param name="id">An ID to uniquely identify the visitor.</param>
        /// <param name="startNode">The start node for the traveral.</param>
        /// <param name="startingPointIndex">The starting point index.</param>
        public OperationNeuralNetworkVisitor(string id, IOperation startNode, int startingPointIndex)
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
            this.operations = new List<IOperation>();
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

        private async Task Traverse(IOperation node, IOperation? fromNode = null)
        {
            if (node == null)
            {
                return;
            }

            if (fromNode != null)
            {
                this.operations.Add(node);

                if (node.VisitedFrom.Contains(fromNode.SpecificId))
                {
                    throw new InvalidOperationException("Node must not be visited twice.");
                }

                node.VisitedFrom.Add(fromNode.SpecificId);
            }

            var dOutput = node.Backward((Matrix)node.BackwardInput);

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
                node.AccumulatedGradients.Add(dOutput);
            }

            if (node.VisitedCount == node.OutputDependencyCount)
            {
                shouldContinue = true;
                if (node.OutputDependencyCount > 1)
                {
                    try
                    {
                        node.SyncSemaphore.Release(node.OutputDependencyCount - 1);
                    }
                    catch (SemaphoreFullException)
                    {
                        Console.WriteLine("Semaphore full exception");
                    }
                }
            }

            node.Lock.ExitWriteLock();

            if (!shouldContinue)
            {
                await node.SyncSemaphore.WaitAsync();

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
                dOutput = GradientUtils.AccumulateBackwardGradients(node.AccumulatedGradients);
            }

            if (node.GradientDestinations != null)
            {
                node.AccumulateGradient(dOutput);
            }

            node.CalculatedGradient = dOutput;

            var adjacentTasks = new List<Task>();
            if (node.HasMultipleInputs)
            {
                Matrix?[] dOutputs = MatrixUtils.Reassemble(dOutput);
                for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
                {
                    IOperation? adjacentOperation = node.BackwardAdjacentOperations[i];
                    if (adjacentOperation != null)
                    {
                        adjacentOperation.BackwardInput = dOutputs[i] ?? throw new InvalidOperationException("The upstream gradient must not be null.");
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
                    IOperation? adjacentOperation = node.BackwardAdjacentOperations[i];
                    if (adjacentOperation != null)
                    {
                        adjacentOperation.BackwardInput = dOutput.Item1 ?? throw new InvalidOperationException("The upstream gradient must not be null.");
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
    }
}

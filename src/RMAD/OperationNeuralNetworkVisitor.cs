//------------------------------------------------------------------------------
// <copyright file="OperationNeuralNetworkVisitor.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Threading;
    using System.Threading.Tasks;

    public class OperationNeuralNetworkVisitor
    {
        private string id;
        private IOperation startNode;
        private int startingPointIndex;
        private bool runInParallel;
        private List<IOperation> operations;

        public OperationNeuralNetworkVisitor(string id, IOperation startNode, int startingPointIndex, bool runInParallel = false)
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
            this.runInParallel = runInParallel;
        }

        public string Id
        {
            get
            {
                return this.id;
            }
        }

        public Task TraverseAsync()
        {
            return this.Traverse(this.startNode);
        }

        private async Task Traverse(IOperation node, IOperation? fromNode = null)
        {
            if (node == null)
            {
                return;
            }

            if (fromNode != null)
            {
                if (node.VisitedFrom.Contains(fromNode.SpecificId))
                {
                    throw new Exception("Node must not be visited twice.");
                }
                node.VisitedFrom.Add(fromNode.SpecificId);
            }

            var dOutput = node.Backward((double[][])node.BackwardInput);

            bool shouldContinue = false;
            node.Initialize(this.startingPointIndex);
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
                        Debug.WriteLine($"Visitor {this.Id} releasing semaphore at node {node.SpecificId} with visited count {node.VisitedCount} {node.OutputDependencyCount}");
                        node.SyncSemaphore.Release(node.OutputDependencyCount - 1);
                        Debug.WriteLine($"Visitor {this.Id} released semaphore at node {node.SpecificId} with visited count {node.VisitedCount} {node.OutputDependencyCount}");
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
                Debug.WriteLine($"Visitor {this.Id} waiting for semaphore at node {node.SpecificId} with visited count {node.VisitedCount} {node.OutputDependencyCount}");
                await node.SyncSemaphore.WaitAsync();
                Debug.WriteLine($"Visitor {this.Id} released from semaphore at node {node.SpecificId} with visited count {node.VisitedCount} {node.OutputDependencyCount}");

                return;
            }

            // Perform gradient accumulation here
            if (node.OutputDependencyCount > 1)
            {

                Debug.WriteLine($"Released to continue {node.SpecificId} {this.startingPointIndex} {node.VisitedCount} {node.OutputDependencyCount}");

                if (node.AccumulatedGradients.Count != node.OutputDependencyCount)
                {
                    throw new Exception("Accumulated gradients count must equal the output dependency count.");
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
            if (node.OperationType == typeof(HadamardProductOperation)
                ||
                node.OperationType == typeof(MatrixMultiplyOperation)
                ||
                node.OperationType == typeof(MatrixAddOperation)
                ||
                node.OperationType == typeof(MatrixAddThreeOperation))
            {
                double[][][] dOutputs = MatrixUtils.Reassemble(dOutput);
                for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
                {
                    IOperation adjacentOperation = node.BackwardAdjacentOperations[i];
                    if (adjacentOperation != null)
                    {
                        adjacentOperation.BackwardInput = dOutputs[i];
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
                        adjacentOperation.BackwardInput = dOutput.Item1;
                        adjacentTasks.Add(this.Traverse(adjacentOperation, node));
                    }
                }
            }

            node.Tasks.AddRange(adjacentTasks);

            await Task.WhenAll(adjacentTasks).ConfigureAwait(false);

            Debug.WriteLine($"Visitor {this.Id} tasks complete at node {node.SpecificId} {node.OutputDependencyCount} {adjacentTasks.Count}");
            this.operations.Add(node);

            node.IsComplete = true;

        }

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
    }
}

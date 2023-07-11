//------------------------------------------------------------------------------
// <copyright file="OperationGraphVisitor.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// The operation graph visitor for setting the backward dependency counts of the computation graph.
    /// </summary>
    public class OperationGraphVisitor
    {
        private readonly string id;
        private readonly IOperationBase startNode;
        private readonly int startingPointIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="OperationGraphVisitor"/> class.
        /// </summary>
        /// <param name="id">A unique ID.</param>
        public OperationGraphVisitor(string id)
        {
            if (string.IsNullOrEmpty(id))
            {
                throw new ArgumentNullException(nameof(id), $"The parameter {nameof(id)} cannot be null or empty.");
            }

            this.id = id;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OperationGraphVisitor"/> class.
        /// </summary>
        /// <param name="id">A unique ID.</param>
        /// <param name="startNode">The start node of the computation graph.</param>
        /// <param name="startingPointIndex">The starting point index.</param>
        public OperationGraphVisitor(string id, IOperationBase startNode, int startingPointIndex)
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
        /// Starts the traversal of the computation graph.
        /// </summary>
        /// <returns>The task.</returns>
        public Task TraverseAsync()
        {
            return this.Traverse(this.startNode);
        }

        /// <summary>
        /// Resets the visited counts of the computation graph.
        /// </summary>
        /// <param name="node">The starting operation.</param>
        /// <param name="returnEarly">If it should return early if the visited count is already 0.</param>
        /// <returns>A task.</returns>
        public async Task ResetVisitedCountsAsync(IOperationBase node, bool returnEarly = true)
        {
            if (node == null)
            {
                return;
            }

            node.Lock.EnterWriteLock();

            if (returnEarly)
            {
                // Return early if VisitedCount is already 0
                if (node.VisitedCount == 0)
                {
                    node.Lock.ExitWriteLock();
                    return;
                }
            }

            node.VisitedCount = 0;
            node.Lock.ExitWriteLock();

            var adjacentTasks = new List<Task>();
            for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
            {
                IOperationBase? adjacentOperation = node.BackwardAdjacentOperations[i];
                if (adjacentOperation != null)
                {
                    adjacentTasks.Add(this.ResetVisitedCountsAsync(adjacentOperation));
                }
            }

            await Task.WhenAll(adjacentTasks);
        }

        private async Task Traverse(IOperationBase node)
        {
            if (node == null)
            {
                return;
            }

            node.InitializeLock();
            node.VisitedCount++;

            node.Lock.EnterWriteLock();
            if (node.BackwardDependencyCounts.Count <= this.startingPointIndex)
            {
                for (int i = node.BackwardDependencyCounts.Count; i <= this.startingPointIndex; ++i)
                {
                    node.BackwardDependencyCounts.Add(0);
                }
            }

            node.BackwardDependencyCounts[this.startingPointIndex]++;
            node.Lock.ExitWriteLock();

            if (node.VisitedCount > 1)
            {
                return;
            }

            var adjacentTasks = new List<Task>();
            for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
            {
                IOperationBase? adjacentOperation = node.BackwardAdjacentOperations[i];
                if (adjacentOperation != null)
                {
                    adjacentTasks.Add(this.Traverse(adjacentOperation));
                }
            }

            await Task.WhenAll(adjacentTasks);
        }
    }
}
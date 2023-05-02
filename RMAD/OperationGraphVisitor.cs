namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    public class OperationGraphVisitor
    {
        private string id;
        private IOperation startNode;
        private int startingPointIndex;

        public OperationGraphVisitor(string id)
        {
            if (string.IsNullOrEmpty(id))
            {
                throw new ArgumentNullException(nameof(id), $"The parameter {nameof(id)} cannot be null or empty.");
            }

            this.id = id;
        }

        public OperationGraphVisitor(string id, IOperation startNode, int startingPointIndex)
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

        public string Id
        {
            get
            {
                return id;
            }
        }

        public Task TraverseAsync()
        {
            return Traverse(startNode);
        }

        private async Task Traverse(IOperation node)
        {
            if (node == null)
            {
                return ;
            }

            Console.WriteLine($"Visitor {this.Id} visiting node {node.SpecificId}");

            node.InitializeLock();
            node.VisitedCount++;

            node.Lock.EnterWriteLock();
            if (node.BackwardDependencyCounts.Count <= startingPointIndex)
            {
                for (int i = node.BackwardDependencyCounts.Count; i <= startingPointIndex; ++i)
                {
                    node.BackwardDependencyCounts.Add(0);
                }
            }
            node.BackwardDependencyCounts[startingPointIndex]++;
            node.Lock.ExitWriteLock();

            if (node.VisitedCount > 1)
            {
                return;
            }

            var adjacentTasks = new List<Task>();
            for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
            {
                IOperation adjacentOperation = node.BackwardAdjacentOperations[i];
                if (adjacentOperation != null)
                {
                    adjacentTasks.Add(Traverse(adjacentOperation));
                }
            }

            await Task.WhenAll(adjacentTasks);
        }

        public async Task ResetVisitedCountsAsync(IOperation node, bool returnEarly = true)
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
            else
            {
                if (node.VisitedCount > 0)
                {

                }
            }

            node.VisitedCount = 0;
            node.Lock.ExitWriteLock();

            var adjacentTasks = new List<Task>();
            for (int i = 0; i < node.BackwardAdjacentOperations.Count; ++i)
            {
                IOperation adjacentOperation = node.BackwardAdjacentOperations[i];
                if (adjacentOperation != null)
                {
                    adjacentTasks.Add(ResetVisitedCountsAsync(adjacentOperation));
                }
            }

            await Task.WhenAll(adjacentTasks);
        }
    }
}
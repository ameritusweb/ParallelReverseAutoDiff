using GradientExplorer.Services;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public class ConcurrentPruner : ConcurrentQueue<ITreeNode>
    {
        private readonly Timer pruneTimer;
        private readonly ILogger logger;
        private bool isTimerActive;

        public ConcurrentPruner(TimeSpan interval, ILogger logger)
        {
            this.logger = logger;
            this.isTimerActive = false;
            pruneTimer = new Timer(PruneNodes, null, interval, interval);
        }

        public void Start()
        {
            if (!isTimerActive)
            {
                // Assuming you want to start the timer with an initial delay of zero
                pruneTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(5));
                isTimerActive = true;
            }
        }

        public void Stop()
        {
            if (isTimerActive)
            {
                pruneTimer.Change(Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
                isTimerActive = false;
            }
        }

        private void PruneNodes(object state)
        {
            foreach (var node in this)
            {
                if (node.MarkForPruning && node.VisitorsCount == 0)
                {
                    // Log the pruning action
                    logger.Log(nameof(MctsEngine), $"Safely pruned node with ID: {node.Id}", Helpers.SeverityType.Information);

                    // Clear children
                    node.Children = new ConcurrentBag<ITreeNode>();

                    // Optionally, remove the node from the queue after pruning
                    TryDequeue(out _);
                }
            }
        }
    }
}

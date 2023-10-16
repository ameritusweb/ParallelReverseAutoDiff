using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public interface ITreeNode
    {
        int Visits { get; set; }
        double Score { get; set; }
        ConcurrentBag<ITreeNode> Children { get; set; } // Using ITreeNode for children
        GameState GameState { get; set; }
        ITreeNode? Parent { get; set; } // Using ITreeNode for parent
        SimplificationAction Action { get; set; }
        bool MarkForPruning { get; set; }
        int VisitorsCount { get; set; }
        bool IsFullyExpanded { get; set; }
        int VisitedForPruning { get; set; }
        string Id { get; }

        void AtomicIncrementVisitedForPruning();
        void AtomicIncrementVisitorsCount();
        void AtomicDecrementVisitorsCount();
        void AtomicIncrementVisits();
        void AtomicAddToScore(double scoreToAdd);
    }
}

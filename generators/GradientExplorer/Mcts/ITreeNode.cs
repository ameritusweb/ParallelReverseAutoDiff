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

        void AtomicIncrementVisits();
        void AtomicAddToScore(double scoreToAdd);
    }
}

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public class TreeNode : ITreeNode
    {
        private int _visits;
        public int Visits
        {
            get
            {
                return _visits;
            }
            set
            {
                _visits = value;
            }
        }

        private int _scaledScore; // Score scaled by 10,000 to preserve four decimal places

        public double Score
        {
            get { return _scaledScore / 10000.0; } // Convert back to double
            set { _scaledScore = (int)(value * 10000); } // Convert to int
        }
        public ConcurrentBag<ITreeNode> Children { get; set; } // Child nodes
        public GameState GameState { get; set; } // Represents the game state at this node
        public ITreeNode? Parent { get; set; }
        public SimplificationAction Action { get; set; }

        public TreeNode(ITreeNode parent = null)
        {
            Children = new ConcurrentBag<ITreeNode>();
            Score = 0;
            Visits = 0;
            Parent = parent;
        }

        // Methods for atomic operations
        public void AtomicIncrementVisits()
        {
            Interlocked.Increment(ref _visits);
        }

        public void AtomicAddToScore(double scoreToAdd)
        {
            int scaledScoreToAdd = (int)(scoreToAdd * 10000); // Scale the score to add
            Interlocked.Add(ref _scaledScore, scaledScoreToAdd); // Add it atomically
        }
    }
}

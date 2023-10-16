using GradientExplorer.Diagram;
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
        public bool MarkForPruning { get; set; } = false;
        public bool IsFullyExpanded { get; set; } = false;
        private int _visitorsCount;
        public int VisitorsCount
        {
            get
            {
                return _visitorsCount;
            }
            set
            {
                _visitorsCount = value;
            }
        }
        private int _visitedForPruning;
        public int VisitedForPruning
        {
            get
            {
                return _visitedForPruning;
            }
            set
            {
                _visitedForPruning = value;
            }
        }

        public string Id { get; private set; }

        public TreeNode(ITreeNode parent = null)
        {
            Children = new ConcurrentBag<ITreeNode>();
            Score = 0;
            Visits = 0;
            Parent = parent;
            Id = DiagramUniqueIDGenerator.Instance.GetNextID();
        }

        // Methods for atomic operations
        public void AtomicIncrementVisitorsCount()
        {
            Interlocked.Increment(ref _visitorsCount);
        }

        public void AtomicIncrementVisitedForPruning()
        {
            Interlocked.Increment(ref _visitedForPruning);
        }

        public void AtomicDecrementVisitorsCount()
        {
            Interlocked.Decrement(ref _visitorsCount);
        }

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

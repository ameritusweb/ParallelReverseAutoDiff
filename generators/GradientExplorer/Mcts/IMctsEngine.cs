using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public interface IMctsEngine
    {
        ITreeNode Root { get; set; }
        int MaxConcurrentRollouts { get; set; }
        double ExplorationConstant { get; set; }
        int RolloutDepth { get; set; }
        ConcurrentEventSystem EventSystem { get; set; }

        /// <summary>
        /// Initializes the root node for the Monte Carlo Tree Search.
        /// </summary>
        void Initialize();

        /// <summary>
        /// Executes the Monte Carlo Tree Search algorithm.
        /// </summary>
        /// <returns>A read-only list of actions representing the best path found.</returns>
        Task<IReadOnlyList<SimplificationAction>> RunMCTS();

        /// <summary>
        /// Selects a node for expansion based on the UCB1 formula.
        /// </summary>
        /// <param name="node">The parent node from which to select a child.</param>
        /// <returns>The selected child node.</returns>
        ITreeNode SelectNode(ITreeNode node);

        /// <summary>
        /// Expands a given node by generating its child nodes.
        /// </summary>
        /// <param name="node">The node to expand.</param>
        /// <param name="currentDepth">The current depth of the node in the tree.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        Task ExpandNodeAsync(ITreeNode node, int currentDepth);

        /// <summary>
        /// Performs a random rollout from a given node and calculates the score.
        /// </summary>
        /// <param name="node">The node from which to start the rollout.</param>
        /// <param name="currentDepth">The current depth of the node in the tree.</param>
        /// <returns>The score obtained from the rollout.</returns>
        double SimulateRandomRollout(ITreeNode node, int currentDepth);

        /// <summary>
        /// Backpropagates the score from a leaf node to the root.
        /// </summary>
        /// <param name="node">The leaf node from which to start backpropagation.</param>
        /// <param name="score">The score to backpropagate.</param>
        void Backpropagate(ITreeNode node, double score);

        /// <summary>
        /// Retrieves the best sequence of actions leading to the given node.
        /// </summary>
        /// <param name="bestNode">The node representing the end of the best path found.</param>
        /// <returns>A read-only list of actions representing the best path.</returns>
        IReadOnlyList<SimplificationAction> GetBestActionSequence(ITreeNode bestNode);

        /// <summary>
        /// Selects the best final node based on specific criteria.
        /// </summary>
        /// <returns>The best final node.</returns>
        ITreeNode SelectBestFinalNode();
    }
}

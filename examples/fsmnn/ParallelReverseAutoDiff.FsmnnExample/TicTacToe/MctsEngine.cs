//------------------------------------------------------------------------------
// <copyright file="MctsEngine.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.TicTacToe
{
    /// <summary>
    /// The MCTS engine.
    /// </summary>
    public class MctsEngine
    {
        private static readonly double C0 = Math.Sqrt(2);

        /// <summary>
        /// Initializes a new instance of the <see cref="MctsEngine"/> class.
        /// </summary>
        /// <param name="initialState">The initial stte.</param>
        public MctsEngine(TicTacToeBoard initialState)
        {
            this.Root = new Node { State = initialState };
        }

        /// <summary>
        /// Gets or sets the root node.
        /// </summary>
        public Node Root { get; set; }

        /// <summary>
        /// Run simulations.
        /// </summary>
        /// <param name="maxSimulations">The max number of simulations.</param>
        public void RunSimulation(int maxSimulations)
        {
            int depth = 1;
            int breadth = 2;
            int simulationCount = 0;

            while (simulationCount < maxSimulations)
            {
                Node nodeToExpand = this.UCTSelectNode(this.Root, simulationCount);
                this.ExpandNode(nodeToExpand, depth, breadth);
                double result = this.Minimax(nodeToExpand, depth, breadth, int.MinValue, int.MaxValue);
                this.Backpropagate(nodeToExpand, result);

                depth++;
                breadth++;
                simulationCount++;
            }
        }

        private Node UCTSelectNode(Node node, int simulationCount)
        {
            double ComputeC()
            {
                // Exponential decay
                return C0 * Math.Pow(0.99, simulationCount);
            }

            double UCT(Node n)
            {
                if (n.Visits == 0)
                {
                    return double.MaxValue;  // encourage unexplored nodes
                }

                double c = ComputeC();

                return (n.Value / n.Visits) + (c * Math.Sqrt(Math.Log(n.Parent.Visits) / n.Visits));
            }

            // If the node has no children, return it
            if (node.Children.Count == 0)
            {
                return node;
            }

            // Otherwise, select the child with the highest UCT value and recurse
            return this.UCTSelectNode(node.Children.OrderByDescending(UCT).First(), simulationCount);
        }

        private void ExpandNode(Node node, int depth, int breadth)
        {
            // Base case: If depth is 0 or the game state is terminal, return
            if (depth == 0 || node.State.IsTerminal())
            {
                return;
            }

            node.Children = new List<Node>();

            // Get all possible moves for the current state
            List<TicTacToeBoard> possibleMoves = node.State.GetPossibleMoves();

            // Limit the number of moves to the specified breadth
            int movesToConsider = Math.Min(breadth, possibleMoves.Count);

            for (int i = 0; i < movesToConsider; i++)
            {
                Node child = new Node
                {
                    State = possibleMoves[i],
                    Parent = node,
                    Children = new List<Node>(),
                    Visits = 0,
                    Value = 0,
                };

                node.Children.Add(child);

                // Recursively expand the child node
                this.ExpandNode(child, depth - 1, breadth);
            }
        }

        private double Minimax(Node node, int depth, int breadth, double alpha, double beta)
        {
            if (depth == 0 || node.State.IsTerminal())
            {
                return this.Evaluate(node.State);
            }

            // Assuming 'X' is the maximizer
            if (node.State.CurrentPlayer == 'X')
            {
                double maxEval = double.NegativeInfinity;
                int counter = 0;

                foreach (var child in node.Children)
                {
                    if (counter++ == breadth)
                    {
                        break; // Only consider 'breadth' number of children
                    }

                    double eval = this.Minimax(child, depth - 1, breadth, alpha, beta);
                    maxEval = Math.Max(maxEval, eval);
                    alpha = Math.Max(alpha, eval);

                    if (beta <= alpha)
                    {
                        break;  // Alpha-beta pruning
                    }
                }

                return maxEval;
            }

            // Assuming 'O' is the minimizer
            else
            {
                double minEval = double.PositiveInfinity;
                int counter = 0;

                foreach (var child in node.Children)
                {
                    if (counter++ == breadth)
                    {
                        break; // Only consider 'breadth' number of children
                    }

                    double eval = this.Minimax(child, depth - 1, breadth, alpha, beta);
                    minEval = Math.Min(minEval, eval);
                    beta = Math.Min(beta, eval);

                    if (beta <= alpha)
                    {
                        break;  // Alpha-beta pruning
                    }
                }

                return minEval;
            }
        }

        private double Evaluate(TicTacToeBoard board)
        {
            // Evaluate the board state
            // Return a high value if 'X' has won, a low value if 'O' has won, and 0 otherwise.
            // Check rows, columns, and diagonals
            for (int i = 0; i < 3; i++)
            {
                // Check rows
                if (board.Board[i, 0] == board.Board[i, 1] && board.Board[i, 1] == board.Board[i, 2])
                {
                    if (board.Board[i, 0] == 'X')
                    {
                        return 10;
                    }
                    else if (board.Board[i, 0] == 'O')
                    {
                        return -10;
                    }
                }

                // Check columns
                if (board.Board[0, i] == board.Board[1, i] && board.Board[1, i] == board.Board[2, i])
                {
                    if (board.Board[0, i] == 'X')
                    {
                        return 10;
                    }
                    else if (board.Board[0, i] == 'O')
                    {
                        return -10;
                    }
                }
            }

            // Check diagonals
            if (board.Board[0, 0] == board.Board[1, 1] && board.Board[1, 1] == board.Board[2, 2])
            {
                if (board.Board[0, 0] == 'X')
                {
                    return 10;
                }
                else if (board.Board[0, 0] == 'O')
                {
                    return -10;
                }
            }

            if (board.Board[0, 2] == board.Board[1, 1] && board.Board[1, 1] == board.Board[2, 0])
            {
                if (board.Board[0, 2] == 'X')
                {
                    return 10;
                }
                else if (board.Board[0, 2] == 'O')
                {
                    return -10;
                }
            }

            // If no one has won, return 0
            return 0;
        }

        private void Backpropagate(Node node, double result, double gamma = 0.9)
        {
            int depth = 0;

            while (node != null)
            {
                // Update the visit count
                node.Visits++;

                // Adjust the result based on the perspective of the player at this node
                if (node.State.CurrentPlayer == 'O')
                {
                    result = -result;
                }

                // Update the value with the discounted result
                node.Value += Math.Pow(gamma, depth) * result;

                // Move to the parent node and increase depth for the next iteration
                node = node.Parent;
                depth++;
            }
        }
    }
}

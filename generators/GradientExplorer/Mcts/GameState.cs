using GradientExplorer.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public class GameState
    {
        public GradientGraph RootGraph { get; set; }
        public double Score { get; set; } // Score based on expression complexity

        // Constructor
        public GameState(GradientGraph rootGraph, double score)
        {
            RootGraph = rootGraph;
            Score = score;
        }

        // Method to apply a simplification rule and return a new GameState
        public GameState ApplyRule(/* Rule parameters */)
        {
            // Implement logic to apply a simplification rule to RootExpression
            // and return a new GameState with the simplified expression and updated score
            return null; // Placeholder
        }

        // Method to evaluate the current game state's score
        public void EvaluateScore()
        {
            // Implement logic to calculate Score based on RootExpression's complexity
            // Score = 1 - sigmoid(complexity)
        }
    }
}

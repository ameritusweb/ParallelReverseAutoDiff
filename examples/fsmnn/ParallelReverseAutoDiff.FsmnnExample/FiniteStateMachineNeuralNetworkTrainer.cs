//------------------------------------------------------------------------------
// <copyright file="FiniteStateMachineNeuralNetworkTrainer.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample
{
    using ParallelReverseAutoDiff.FsmnnExample.Amaze;
    using ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A finite state machine neural network trainer.
    /// </summary>
    public class FiniteStateMachineNeuralNetworkTrainer
    {
        private bool switchGradients = false;

        /// <summary>
        /// Train the network.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Train()
        {
            CudaBlas.Instance.Initialize();
            try
            {
                MazeMaker makeMaker = new MazeMaker();
                var maze = makeMaker.CreateMaze(10);
                var maze2 = makeMaker.ChangePath(maze);
                FiniteStateMachineTraversalNeuralNetwork traversalNetwork = new FiniteStateMachineTraversalNeuralNetwork(maze, 6, maze.NumIndices, maze.AlphabetSize, 40, 4, 4, 0.000000001d, 4d);
                await traversalNetwork.Initialize();

                for (int i = 0; i < 10000; ++i)
                {
                    if (i % 5 == 4)
                    {
                        traversalNetwork.Reinitialize(traversalNetwork.Maze == maze ? maze2 : maze);
                        this.switchGradients = !this.switchGradients;
                        Console.WriteLine("Reinit");
                    }

                    await this.RunIteration(traversalNetwork, traversalNetwork.Maze == maze ? 0.2d : 0.25d);
                }
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        /// <summary>
        /// Runs an iteration of training.
        /// </summary>
        /// <param name="traversalNetwork">The traversal network.</param>
        /// <param name="trueMaxMin">The true max min.</param>
        /// <returns>The task.</returns>
        public async Task RunIteration(FiniteStateMachineTraversalNeuralNetwork traversalNetwork, double trueMaxMin)
        {
            var (initialGradient, initialDist, initialDiff, initialOutput) = traversalNetwork.Forward2(trueMaxMin);
            await traversalNetwork.Backward(initialGradient);

            await this.TryUpdateWithLearningRate(traversalNetwork, initialDiff, 0.000001d, trueMaxMin, initialOutput);

            await traversalNetwork.Reset();
        }

        private async Task TryUpdateWithLearningRate(FiniteStateMachineTraversalNeuralNetwork traversalNetwork, double prevDiff, double learningRate, double trueMaxMin, Matrix initialOutput)
        {
            // Base case: if learning rate is too small, terminate recursion
            if (learningRate < 1e-9)
            {
                return;
            }

            traversalNetwork.AdjustLearningRate(learningRate);
            traversalNetwork.ApplyGradients(this.switchGradients);

            var (gradient, dist, diff, output) = traversalNetwork.Forward2(trueMaxMin);

            if (diff > prevDiff || !this.HaveSameOrdering(initialOutput[0], output[0], learningRate))
            {
                // If performance didn't improve, revert the update and try again with a reduced learning rate
                traversalNetwork.RevertUpdate(this.switchGradients);
                await this.TryUpdateWithLearningRate(traversalNetwork, prevDiff, learningRate * 0.1, trueMaxMin, initialOutput);
            }
        }

        private bool HaveSameOrdering(double[] vec1, double[] vec2, double learningRate, int tolerance = 3)
        {
            if (vec1.Length != vec2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            if (learningRate >= 0.000001)
            {
                tolerance = 6;
            }

            // Get the sorted indices for both vectors
            int[] sortedIndicesVec1 = vec1.Select((value, index) => new { value, index })
                                          .OrderBy(item => item.value)
                                          .Select(item => item.index)
                                          .ToArray();

            int[] sortedIndicesVec2 = vec2.Select((value, index) => new { value, index })
                                          .OrderBy(item => item.value)
                                          .Select(item => item.index)
                                          .ToArray();

            // Count the differing positions
            int differingPositions = sortedIndicesVec1.Zip(sortedIndicesVec2, (a, b) => a == b ? 0 : 1).Sum();

            // Check if the number of differing positions is within the tolerance
            var isGood = differingPositions <= tolerance;
            if (isGood)
            {
            }

            return isGood;
        }
    }
}

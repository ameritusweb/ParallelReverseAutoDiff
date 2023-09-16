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
                var maze2 = makeMaker.CreateMaze(10);
                FiniteStateMachineTraversalNeuralNetwork traversalNetwork = new FiniteStateMachineTraversalNeuralNetwork(maze, 6, maze.NumIndices, maze.AlphabetSize, 40, 4, 4, 0.000000001d, 4d);
                await traversalNetwork.Initialize();

                for (int i = 0; i < 10000; ++i)
                {
                    if (i % 20 == 19)
                    {
                        traversalNetwork.Reinitialize(traversalNetwork.Maze == maze ? maze2 : maze);
                    }

                    await this.RunIteration(traversalNetwork);
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
        /// <returns>The task.</returns>
        public async Task RunIteration(FiniteStateMachineTraversalNeuralNetwork traversalNetwork)
        {
            var (gradient, dist, diff) = traversalNetwork.Forward2();
            await traversalNetwork.Backward(gradient);
            traversalNetwork.AdjustLearningRate(0.000001d);
            traversalNetwork.ApplyGradients();
            var (gradient1, dist1, diff1) = traversalNetwork.Forward2();
            if (diff1 > diff)
            {
                traversalNetwork.RevertUpdate();
                traversalNetwork.AdjustLearningRate(0.0000001d);
                traversalNetwork.ApplyGradients();
                var (gradient2, dist2, diff2) = traversalNetwork.Forward2();
                if (diff2 > diff)
                {
                    traversalNetwork.RevertUpdate();
                    traversalNetwork.AdjustLearningRate(0.00000001d);
                    traversalNetwork.ApplyGradients();
                    var (gradient3, dist3, diff3) = traversalNetwork.Forward2();
                    if (diff3 > diff)
                    {
                        traversalNetwork.RevertUpdate();
                    }
                }
            }

            await traversalNetwork.Reset();
        }
    }
}

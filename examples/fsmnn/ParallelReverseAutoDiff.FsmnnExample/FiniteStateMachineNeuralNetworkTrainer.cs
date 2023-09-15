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
                FiniteStateMachineTraversalNeuralNetwork traversalNetwork = new FiniteStateMachineTraversalNeuralNetwork(maze, 6, maze.NumIndices, maze.AlphabetSize, 40, 4, 4, 0.000000001d, 4d);
                await traversalNetwork.Initialize();

                for (int i = 0; i < 10000; ++i)
                {
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
            var gradient = traversalNetwork.Forward2();
            await traversalNetwork.Backward(gradient);
            traversalNetwork.ApplyGradients();
            await traversalNetwork.Reset();
        }
    }
}

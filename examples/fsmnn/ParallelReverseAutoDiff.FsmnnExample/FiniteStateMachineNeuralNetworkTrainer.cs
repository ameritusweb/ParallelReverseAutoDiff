//------------------------------------------------------------------------------
// <copyright file="FiniteStateMachineNeuralNetworkTrainer.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample
{
    using ParallelReverseAutoDiff.FsmnnExample.Amaze;
    using ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork;

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
            MazeMaker makeMaker = new MazeMaker();
            var maze = makeMaker.CreateMaze(10);
            FiniteStateMachineTraversalNeuralNetwork traversalNetwork = new FiniteStateMachineTraversalNeuralNetwork(maze, 6, maze.NumIndices, maze.AlphabetSize, 10, 2, 2, 0.001d, 4d);

            for (int i = 0; i < 100; ++i)
            {
                await this.RunIteration(traversalNetwork);
            }
        }

        /// <summary>
        /// Runs an iteration of training.
        /// </summary>
        /// <param name="traversalNetwork">The traversal network.</param>
        /// <returns>The task.</returns>
        public async Task RunIteration(FiniteStateMachineTraversalNeuralNetwork traversalNetwork)
        {
            await traversalNetwork.Initialize();
            var gradient = traversalNetwork.Forward();
            await traversalNetwork.Backward(gradient);
            traversalNetwork.ApplyGradients();
            await traversalNetwork.Reset();
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="FiniteStateMachineTraversalNeuralNetwork.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork
{
    using System;
    using System.IO;
    using ParallelReverseAutoDiff.FsmnnExample.Amaze;
    using ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD;
    using ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork.Embedding;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Finite state machine traversal neural network.
    /// </summary>
    public class FiniteStateMachineTraversalNeuralNetwork
    {
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numIndices;
        private readonly int numLayers;
        private readonly int numQueries;
        private readonly int alphabetSize;
        private readonly int embeddingSize;
        private readonly double learningRate;
        private readonly double clipValue;
        private EmbeddingNeuralNetwork embeddingNeuralNetwork;

        private DirectedAdamOptimizer optimizer;

        private Maze maze;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="FiniteStateMachineTraversalNeuralNetwork"/> class.
        /// </summary>
        /// <param name="maze">The maze.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numIndices">The number of indices.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <param name="embeddingSize">The embedding size.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public FiniteStateMachineTraversalNeuralNetwork(Maze maze, int numNodes, int numIndices, int alphabetSize, int embeddingSize, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.maze = maze;
            this.numFeatures = numIndices * embeddingSize;
            this.alphabetSize = alphabetSize;
            this.embeddingSize = embeddingSize;
            this.numNodes = numNodes;
            this.numIndices = numIndices;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.embeddingNeuralNetwork = new EmbeddingNeuralNetwork(this.numLayers, this.numQueries, this.numNodes, this.numFeatures, this.numIndices, this.alphabetSize, this.embeddingSize, this.learningRate, this.clipValue);
            this.optimizer = new DirectedAdamOptimizer(this.embeddingNeuralNetwork, false);
        }

        /// <summary>
        /// Gets the maze.
        /// </summary>
        public Maze Maze
        {
            get
            {
                return this.maze;
            }
        }

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.embeddingNeuralNetwork.Initialize();
            this.embeddingNeuralNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.embeddingNeuralNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Reinitialize with new maze.
        /// </summary>
        /// <param name="maze">The maze.</param>
        public void Reinitialize(Maze maze)
        {
            this.maze = maze;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 1;
            var model = new EmbeddingNeuralNetwork(this.numLayers, this.numQueries, this.numNodes, this.numFeatures, this.numIndices, this.alphabetSize, this.embeddingSize, this.learningRate, this.clipValue);
            model.Parameters.AdamIteration = initialAdamIteration;
            this.embeddingNeuralNetwork = model;
            this.optimizer = new DirectedAdamOptimizer(this.embeddingNeuralNetwork, false);
            await this.embeddingNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.embeddingNeuralNetwork.ModelLayers).ToList();

            // this.SaveWeights();
            // this.ApplyWeights();
        }

        /// <summary>
        /// Save the weights to the save path.
        /// </summary>
        public void SaveWeights()
        {
            Guid guid = Guid.NewGuid();
            var dir = $"E:\\store\\{guid}_{this.embeddingNeuralNetwork.Parameters.AdamIteration}";
            Directory.CreateDirectory(dir);
            int index = 0;
            foreach (var modelLayer in this.modelLayers)
            {
                modelLayer.SaveWeightsAndMomentsBinary(new FileInfo($"{dir}\\layer{index}"));
                index++;
            }
        }

        /// <summary>
        /// Apply the weights from the save path.
        /// </summary>
        public void ApplyWeights()
        {
            var guid = "a9898d81-8c2e-4626-abe6-6267babb5e2f_1";
            var dir = $"E:\\store\\{guid}";
            for (int i = 0; i < this.modelLayers.Count; ++i)
            {
                var modelLayer = this.modelLayers[i];
                var file = new FileInfo($"{dir}\\layer{i}");
                modelLayer.LoadWeightsAndMomentsBinary(file);
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
            }
        }

        /// <summary>
        /// Apply the gradients to update the weights.
        /// </summary>
        /// <param name="switchGradients">The true max min.</param>
        public void ApplyGradients(bool switchGradients)
        {
            var clipper = this.embeddingNeuralNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.optimizer;
            adamOptimizer.SwitchGradients = switchGradients;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Reverts the weight update.
        /// </summary>
        /// <param name="switchGradients">The true max min.</param>
        public void RevertUpdate(bool switchGradients)
        {
            var adamOptimizer = this.optimizer;
            adamOptimizer.SwitchGradients = switchGradients;
            adamOptimizer.Revert(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public Matrix Forward()
        {
            var embeddingNet = this.embeddingNeuralNetwork;
            embeddingNet.NumPath = this.maze.MazePath.MazeNodes.Length;
            embeddingNet.InitializeState();
            var indices = this.maze.ToIndices();
            embeddingNet.AutomaticForwardPropagate(indices);
            var output = embeddingNet.Output;
            Console.WriteLine(output[0][0] + " " + output[0][1] + " " + output[0][2] + " " + output[0][output.Cols - 1]);
            CategoricalCrossEntropyLossOperation lossOperation = new CategoricalCrossEntropyLossOperation();
            lossOperation.Forward(output, this.maze.ToTrueLabel(output.Cols));
            var gradientOfLoss = lossOperation.Backward();

            return gradientOfLoss;
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <param name="trueMaxMin">The true max min.</param>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix Gradient, double Dist, double Diff, Matrix Output) Forward2(double trueMaxMin)
        {
            var embeddingNet = this.embeddingNeuralNetwork;
            embeddingNet.NumPath = this.maze.MazePath.MazeNodes.Length;
            embeddingNet.InitializeState();
            var indices = this.maze.ToIndices();
            embeddingNet.AutomaticForwardPropagate(indices);
            var output = embeddingNet.Output;
            Console.WriteLine(output[0][0] + " " + output[0][1] + " " + output[0][2] + " " + output[0][3] + " " + output[0][4] + " " + output[0][5]);
            VarianceAlphaSearchLossOperation lossOperation = new VarianceAlphaSearchLossOperation();
            var dist = lossOperation.Forward(output, 0.004d, trueMaxMin);
            var gradientOfLoss = lossOperation.Backward();

            return (gradientOfLoss, dist, Math.Abs(dist - trueMaxMin), new Matrix(output.ToArray()));
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task Backward(Matrix gradientOfLossWrtOutput)
        {
            await this.embeddingNeuralNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
        }
    }
}

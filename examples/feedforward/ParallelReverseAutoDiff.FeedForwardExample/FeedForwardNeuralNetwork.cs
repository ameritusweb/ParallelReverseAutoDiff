//------------------------------------------------------------------------------
// <copyright file="FeedForwardNeuralNetwork.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A feed forward neural network.
    /// </summary>
    public class FeedForwardNeuralNetwork
    {
        /// <summary>
        /// Gets or sets the embedding layer.
        /// </summary>
        public EmbeddingLayer EmbeddingLayer { get; set; } = new EmbeddingLayer();

        /// <summary>
        /// Gets or sets the hidden layer.
        /// </summary>
        public HiddenLayer HiddenLayer { get; set; } = new HiddenLayer();

        /// <summary>
        /// Gets or sets the output layer.
        /// </summary>
        public OutputLayer OutputLayer { get; set; } = new OutputLayer();
    }
}

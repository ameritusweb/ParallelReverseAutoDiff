//------------------------------------------------------------------------------
// <copyright file="NeuralNetworkUtilities.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The utilities for a neural network.
    /// </summary>
    public class NeuralNetworkUtilities
    {
        private NeuralNetwork network;
        private AdamOptimizer adamOptimizer;
        private GradientClipper gradientClipper;

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetworkUtilities"/> class.
        /// </summary>
        /// <param name="network">The neural network.</param>
        public NeuralNetworkUtilities(NeuralNetwork network)
        {
            this.network = network;
        }

        /// <summary>
        /// Gets the Adam optimizer.
        /// </summary>
        public AdamOptimizer AdamOptimizer
        {
            get
            {
                return this.adamOptimizer ??= new AdamOptimizer(this.network);
            }
        }

        /// <summary>
        /// Gets the gradient clipper.
        /// </summary>
        public GradientClipper GradientClipper
        {
            get
            {
                return this.gradientClipper ??= new GradientClipper(this.network);
            }
        }
    }
}

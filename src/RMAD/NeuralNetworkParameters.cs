﻿//------------------------------------------------------------------------------
// <copyright file="NeuralNetworkParameters.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The parameters for a neural network.
    /// </summary>
    public class NeuralNetworkParameters
    {
        /// <summary>
        /// Gets or sets the dropout rate for the apply dropout operation.
        /// </summary>
        public double DropoutRate { get; set; } = 0.01d;

        /// <summary>
        /// Gets or sets the discount factor.
        /// </summary>
        public double DiscountFactor { get; set; } = 0.99d;

        /// <summary>
        /// Gets or sets the alpha value for the LeakyReLU operation.
        /// </summary>
        public double LeakyReLUAlpha { get; set; } = 0.01d;

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double LearningRate { get; set; } = 0.001d;

        /// <summary>
        /// Gets or sets the number of time steps.
        /// </summary>
        public int NumTimeSteps { get; set; }

        /// <summary>
        /// Gets or sets the input sequence.
        /// </summary>
        public Matrix[] InputSequence { get; set; }

        /// <summary>
        /// Gets or sets the rewards for policy gradient optimization.
        /// </summary>
        public List<double> Rewards { get; set; }

        /// <summary>
        /// Gets or sets the chosen actions for policy gradient optimization.
        /// </summary>
        public List<Matrix> ChosenActions { get; set; }
    }
}
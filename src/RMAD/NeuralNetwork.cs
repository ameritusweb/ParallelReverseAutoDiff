//------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// The base class for a neural network.
    /// </summary>
    public abstract class NeuralNetwork
    {
        /// <summary>
        /// Gets or sets the dropout rate for the apply dropout operation.
        /// </summary>
        protected double DropoutRate { get; set; } = 0.01d;

        /// <summary>
        /// Gets or sets the discount factor.
        /// </summary>
        protected double DiscountFactor { get; set; } = 0.99d;

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        protected double LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the number of time steps.
        /// </summary>
        protected int NumTimeSteps { get; set; }

        /// <summary>
        /// Gets or sets the input sequence.
        /// </summary>
        protected Matrix[] InputSequence { get; set; }

        /// <summary>
        /// Gets or sets the rewards for policy gradient optimization.
        /// </summary>
        protected List<double> Rewards { get; set; }

        /// <summary>
        /// Gets or sets the chosen actions for policy gradient optimization.
        /// </summary>
        protected List<Matrix> ChosenActions { get; set; }

        /// <summary>
        /// Gets the dropout rate for the apply dropout operation.
        /// </summary>
        /// <returns>The dropout rate.</returns>
        public virtual double GetDropoutRate()
        {
            return this.DropoutRate;
        }

        /// <summary>
        /// Gets the discount factor.
        /// </summary>
        /// <returns>The discount factor.</returns>
        public virtual double GetDiscountFactor()
        {
            return this.DiscountFactor;
        }

        /// <summary>
        /// Gets the number of time steps.
        /// </summary>
        /// <returns>The number of tine steps.</returns>
        public virtual int GetNumTimeSteps()
        {
            return this.NumTimeSteps;
        }

        /// <summary>
        /// Gets the learning rate.
        /// </summary>
        /// <returns>The learning rate.</returns>
        public virtual double GetLearningRate()
        {
            return this.LearningRate;
        }

        /// <summary>
        /// Gets the rewards for policy gradient optimization.
        /// </summary>
        /// <returns>The rewards.</returns>
        public virtual List<double> GetRewards()
        {
            return this.Rewards;
        }

        /// <summary>
        /// Gets the input sequence.
        /// </summary>
        /// <returns>The input sequence.</returns>
        public virtual Matrix[] GetInputSequence()
        {
            return this.InputSequence;
        }

        /// <summary>
        /// Gets the chosen actions for policy gradient optimization.
        /// </summary>
        /// <returns>The chosen actions.</returns>
        public virtual List<Matrix> GetChosenActions()
        {
            return this.ChosenActions;
        }
    }
}

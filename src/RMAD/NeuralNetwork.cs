//------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;

    public abstract class NeuralNetwork
    {
        protected readonly Func<double, double> sigmoid = (x) => 1 / (1 + Math.Exp(-x));
        protected readonly Func<double, double> tanh = (d) => Math.Sinh(d) / Math.Cosh(d);

        protected double dropoutRate = 0.01d;

        protected double discountFactor = 0.99d;

        protected double learningRate;

        protected int numTimeSteps;

        protected double[][][] inputSequence;

        protected List<double> rewards;

        protected List<double[][]> chosenActions;

        public virtual double GetDropoutRate()
        {
            return this.dropoutRate;
        }

        public virtual double GetDiscountFactor()
        {
            return this.discountFactor;
        }

        public virtual int GetNumTimeSteps()
        {
            return this.numTimeSteps;
        }

        public virtual double GetLearningRate()
        {
            return this.learningRate;
        }

        public virtual List<double> GetRewards()
        {
            return this.rewards;
        }

        public virtual double[][][] GetInputSequence()
        {
            return this.inputSequence;
        }

        public virtual List<double[][]> GetChosenActions()
        {
            return this.chosenActions;
        }
    }
}

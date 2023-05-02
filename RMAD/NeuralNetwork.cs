namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;

    public abstract class NeuralNetwork
    {
        protected readonly Func<double, double> sigmoid = (x) => 1 / (1 + Math.Exp(-x));
        protected readonly Func<double, double> tanh = (d) => Math.Sinh(d) / Math.Cosh(d);
        protected readonly double dropoutRate = 0.01d;
        protected readonly double discountFactor = 0.99d;

        protected int learningRate;

        protected int numTimeSteps;

        protected double[][][] inputSequence;

        protected List<double> rewards;

        protected List<double[][]> chosenActions;

        public virtual double GetDropoutRate()
        {
            return dropoutRate;
        }

        public virtual double GetDiscountFactor()
        {
            return discountFactor;
        }

        public virtual int GetNumTimeSteps()
        {
            return numTimeSteps;
        }

        public virtual double GetLearningRate()
        {
            return learningRate;
        }

        public virtual List<double> GetRewards()
        {
            return rewards;
        }

        public virtual double[][][] GetInputSequence()
        {
            return inputSequence;
        }

        public virtual List<double[][]> GetChosenActions()
        {
            return chosenActions;
        }
    }
}

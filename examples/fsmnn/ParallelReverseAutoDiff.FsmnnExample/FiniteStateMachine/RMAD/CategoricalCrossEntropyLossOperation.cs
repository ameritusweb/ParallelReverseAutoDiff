//------------------------------------------------------------------------------
// <copyright file="CategoricalCrossEntropyLossOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Categorical cross-entropy loss operation.
    /// </summary>
    public class CategoricalCrossEntropyLossOperation
    {
        private Matrix predicted;
        private Matrix trueLabel;

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="predicted">1xN matrix of predicted probabilities.</param>
        /// <param name="trueLabel">1xN matrix of true labels (one-hot encoded).</param>
        /// <returns>The computed loss.</returns>
        public double Forward(Matrix predicted, Matrix trueLabel)
        {
            if (predicted.Cols != trueLabel.Cols)
            {
                throw new ArgumentException("Predicted and trueLabel matrices must have the same dimensions.");
            }

            this.predicted = predicted;
            this.trueLabel = trueLabel;

            double loss = 0;
            for (int i = 0; i < predicted.Cols; i++)
            {
                loss -= trueLabel[0][i] * Math.Log(predicted[0][i]);
            }

            return loss;
        }

        /// <summary>
        /// Computes the gradient with respect to the predicted probabilities.
        /// </summary>
        /// <returns>1xN matrix of gradients.</returns>
        public Matrix Backward()
        {
            Matrix dOutput = new Matrix(1, this.predicted.Cols);

            for (int i = 0; i < this.predicted.Cols; i++)
            {
                dOutput[0][i] = -(this.trueLabel[0][i] / this.predicted[0][i]);
            }

            return dOutput;
        }
    }
}

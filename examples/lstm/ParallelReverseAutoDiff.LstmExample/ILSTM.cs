//------------------------------------------------------------------------------
// <copyright file="ILSTM.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An interface for a Long Short-Term Memory (LSTM) neural network.
    /// </summary>
    public interface ILSTM
    {
        /// <summary>
        /// Gets the name of the neural network used for debugging purposes.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Get the output of the LSTM for the given inputs.
        /// </summary>
        /// <param name="inputs">The inputs where the first dimension is the time dimension.</param>
        /// <returns>The output of the LSTM per time step.</returns>
        double[] GetOutput(Matrix[] inputs);

        /// <summary>
        /// Run the forward and backward passes of the LSTM for the given inputs and chosen actions.
        /// </summary>
        /// <param name="inputs">The input data of the LSTM.</param>
        /// <param name="chosenActions">The actions chosen by the agent.</param>
        /// <param name="rewards">The positive or negative rewards received for the chosen action.</param>
        /// <param name="iterationIndex">The iteration index used for Adam optimization.</param>
        /// <param name="doNotUpdate">Whether or not to update the network's parameters.</param>
        /// <returns>A task with the result of the async operation.</returns>
        Task Optimize(Matrix[] inputs, List<Matrix> chosenActions, List<double> rewards, int iterationIndex, bool doNotUpdate = false);

        /// <summary>
        /// Saves the parameters of the neural network to the given path.
        /// </summary>
        /// <param name="path">The path to store the parameters of the newral network.</param>
        void SaveModel(string path);
    }
}
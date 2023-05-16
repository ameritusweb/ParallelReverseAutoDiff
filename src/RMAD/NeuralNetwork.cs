//------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The base class for a neural network.
    /// </summary>
    public abstract class NeuralNetwork
    {
        private NeuralNetworkParameters parameters;

        /// <summary>
        /// Gets the parameters for the neural network.
        /// </summary>
        public NeuralNetworkParameters Parameters
        {
            get
            {
                return this.parameters ??= new NeuralNetworkParameters();
            }
        }

        /// <summary>
        /// Lookup the parameters for the operation.
        /// </summary>
        /// <param name="op">The operation to lookup.</param>
        /// <returns>The parameters.</returns>
        protected virtual object[] LookupParameters(IOperationBase op)
        {
            object[] operationParameters = op.Parameters;
            object[] parametersToReturn = new object[operationParameters.Length];
            for (int j = 0; j < operationParameters.Length; ++j)
            {
                if (operationParameters[j] is IOperationBase)
                {
                    parametersToReturn[j] = ((IOperationBase)operationParameters[j]).GetOutput();
                }
                else
                {
                    parametersToReturn[j] = operationParameters[j];
                }
            }

            return parametersToReturn;
        }
    }
}

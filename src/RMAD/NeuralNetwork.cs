//------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The base class for a neural network.
    /// </summary>
    public abstract class NeuralNetwork
    {
        private NeuralNetworkParameters parameters;
        private NeuralNetworkUtilities utilities;

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
        /// Gets the utilities for the neural network.
        /// </summary>
        public NeuralNetworkUtilities Utilities
        {
            get
            {
                return this.utilities ??= new NeuralNetworkUtilities(this);
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
                if (operationParameters[j] is IOperation)
                {
                    parametersToReturn[j] = ((IOperation)operationParameters[j]).GetOutput();
                }
                else if (operationParameters[j] is IDeepOperation)
                {
                    parametersToReturn[j] = ((IDeepOperation)operationParameters[j]).GetDeepOutput();
                }
                else if (operationParameters[j] is IBatchOperation)
                {
                    parametersToReturn[j] = ((IBatchOperation)operationParameters[j]).GetDeepOutput();
                }
                else if (operationParameters[j] is IOperationBase[] parameters)
                {
                    List<object> parametersList = new List<object>();
                    for (int k = 0; k < parameters.Length; ++k)
                    {
                        if (parameters[k] is IOperation)
                        {
                            parametersList.Add(((IOperation)parameters[k]).GetOutput());
                        }
                        else if (parameters[k] is IDeepOperation)
                        {
                            parametersList.Add(((IDeepOperation)parameters[k]).GetDeepOutput());
                        }
                        else if (parameters[k] is IBatchOperation)
                        {
                            parametersList.Add(((IBatchOperation)parameters[k]).GetDeepOutput());
                        }
                        else
                        {
                            parametersList.Add(parameters[k]);
                        }
                    }

                    parametersToReturn[j] = parametersList.ToArray();
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

//------------------------------------------------------------------------------
// <copyright file="Program.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
// See https://aka.ms/new-console-template for more information
using ParallelReverseAutoDiff.FeedForwardExample;
using ParallelReverseAutoDiff.RMAD;

FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(100, 1000, 1, 3, 0.001d, null);
await neuralNetwork.Initialize();
for (int i = 0; i < 100; i++)
{
    Matrix input = new Matrix(100, 1);
    Matrix target = new Matrix(1, 1);
    target[0][0] = 0.5d;
    CudaBlas.Instance.Initialize();
    await Task.Delay(5000);
    try
    {
        await neuralNetwork.Optimize(input, target, i, null);
    }
    finally
    {
        CudaBlas.Instance.Dispose();
    }
}
//------------------------------------------------------------------------------
// <copyright file="Program.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
// See https://aka.ms/new-console-template for more information
using ParallelReverseAutoDiff.FsmnnExample;

Console.WriteLine("Hello, World!");

FiniteStateMachineNeuralNetworkTrainer fsmTrainer = new FiniteStateMachineNeuralNetworkTrainer();
await fsmTrainer.Train();
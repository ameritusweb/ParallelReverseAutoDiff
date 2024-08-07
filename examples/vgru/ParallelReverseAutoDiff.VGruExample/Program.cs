﻿//------------------------------------------------------------------------------
// <copyright file="Program.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    /// <summary>
    /// Entry point for the application.
    /// </summary>
    internal class Program
    {
        private static void Main(string[] args)
        {
            MazeNetTrainer trainer = new MazeNetTrainer();
            Task.Run(async () => await trainer.Train()).Wait();
        }
    }
}
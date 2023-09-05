//------------------------------------------------------------------------------
// <copyright file="Program.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
// See https://aka.ms/new-console-template for more information
using ParallelReverseAutoDiff.FsmnnExample.Amaze;

Console.WriteLine("Hello, World!");

int[] indices = CubeSplitter.FindQuadrantIndices(new Point3d(1, 1, 1), 4);
Console.WriteLine(indices[0]);
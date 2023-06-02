//------------------------------------------------------------------------------
// <copyright file="Program.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
// See https://aka.ms/new-console-template for more information
using ParallelReverseAutoDiff.GnnExample;

Console.WriteLine("Hello, World!");

GameGenerator gameGenerator = new GameGenerator();
var pgn = gameGenerator.GenerateBoth();
Console.WriteLine(pgn);
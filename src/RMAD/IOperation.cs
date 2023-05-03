//------------------------------------------------------------------------------
// <copyright file="IOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;

    public interface IOperation
    {
        int TimeStepIndex { get; set; }

        int LayerIndex { get; set; }

        Type OperationType { get; set; }

        bool HasNext { get; }

        IOperation Next { get; set; }

        bool IsComplete { get; set; }

        string Id { get; set; }

        string SpecificId { get; set; }

        double[][] GetOutput();

        void Reset();

        (double[][]?, double[][]?) Backward(double[][] dOutput);

        object[] GradientDestinations { get; set; }

        void AccumulateGradient((double[][]?, double[][]?) dOutput);

        string ResultToName { get; set; }

        void ResultTo(Func<int, int, object> func);

        void Initialize(int startingPointIndex);

        void InitializeLock();

        void InitializeSyncSemaphore();

        object[] Parameters { get; set; }

        List<Task> Tasks { get; set; }

        List<string> Inputs { get; set; }

        List<string> Outputs { get; set; }

        object BackwardInput { get; set; }

        List<IOperation> BackwardAdjacentOperations { get; set; }

        List<int> BackwardDependencyCounts { get; set; }

        List<List<string>> BackwardDependencies { get; set; }

        List<string> VisitedFrom { get; set; }

        List<(double[][]?, double[][]?)> AccumulatedGradients { get; set; }

        (double[][]?, double[][]?) CalculatedGradient { get; set; }

        int OutputDependencyCount { get; set; }

        int VisitedCount { get; set; }

        ReaderWriterLockSlim Lock { get; set; }

        SemaphoreSlim SyncSemaphore { get; set; }
    }
}

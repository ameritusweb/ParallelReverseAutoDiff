﻿//------------------------------------------------------------------------------
// <copyright file="IOperationBase.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// Define the abstract base class for all operations.
    /// </summary>
    public interface IOperationBase
    {
        /// <summary>
        /// Gets or sets the time step of the current operation.
        /// </summary>
        int TimeStepIndex { get; set; }

        /// <summary>
        /// Gets or sets the layer index of the current operation.
        /// </summary>
        int LayerIndex { get; set; }

        /// <summary>
        /// Gets or sets the type of operation (e.g. MatrixMultiplyOperation, MatrixAddOperation).
        /// </summary>
        Type OperationType { get; set; }

        /// <summary>
        /// Gets a value indicating whether there's a next operation in the sequence.
        /// </summary>
        bool HasNext { get; }

        /// <summary>
        /// Gets or sets the reference to the next operation in the sequence.
        /// </summary>
        IOperationBase Next { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the operation is complete.
        /// </summary>
        bool IsComplete { get; set; }

        /// <summary>
        /// Gets or sets the operation ID.
        /// </summary>
        string Id { get; set; }

        /// <summary>
        /// Gets or sets the specific ID of the operation.
        /// </summary>
        string SpecificId { get; set; }

        /// <summary>
        /// Gets or sets the nested specific ID of the operation.
        /// </summary>
        string NestedSpecificId { get; set; }

        /// <summary>
        /// Gets or sets the parameters to the Forward function for this operation.
        /// </summary>
        object[] Parameters { get; set; }

        /// <summary>
        /// Gets or sets the backward tasks running for this operation.
        /// </summary>
        List<Task> Tasks { get; set; }

        /// <summary>
        /// Gets or sets the specific ID of the operations whose outputs are the inputs to the Forward function for this operation.
        /// </summary>
        List<string> Inputs { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to switch the first two dimensions of the input.
        /// </summary>
        bool SwitchFirstTwoDimensions { get; set; }

        /// <summary>
        /// Gets or sets the specific ID of the operations who take in this operation's output as input.
        /// </summary>
        List<string> Outputs { get; set; }

        /// <summary>
        /// Gets or sets the input to the Backward function for this operation.
        /// </summary>
        object BackwardInput { get; set; }

        /// <summary>
        /// Gets or sets the operations that are next when traversing the computational graph via the backward pass.
        /// </summary>
        List<IOperationBase?> BackwardAdjacentOperations { get; set; }

        /// <summary>
        /// Gets or sets the number of operations that take this operation's output as input based on the timestep that you start at when doing the backward pass.
        /// </summary>
        List<int> BackwardDependencyCounts { get; set; }

        /// <summary>
        /// Gets or sets the backward dependencies for this operation.
        /// </summary>
        List<List<string>> BackwardDependencies { get; set; }

        /// <summary>
        /// Gets or sets which node this node was visited from.
        /// </summary>
        List<string> VisitedFrom { get; set; }

        /// <summary>
        /// Gets or sets the accumulated gradients from all output dependent operations.
        /// </summary>
        List<BackwardResult> AccumulatedGradients { get; set; }

        /// <summary>
        /// Gets or sets the accumulated gradients from all backward passes through this operation node.
        /// </summary>
        object?[] CalculatedGradient { get; set; }

        /// <summary>
        /// Gets or sets, for the current backward pass, the number of operations that take this operation's output as input.
        /// </summary>
        int OutputDependencyCount { get; set; }

        /// <summary>
        /// Gets or sets the number of times this operation node has been visited during a specific pass.
        /// </summary>
        int VisitedCount { get; set; }

        /// <summary>
        /// Gets or sets a lock to handle issues that arise from concurrent access to shared resources.
        /// </summary>
        ReaderWriterLockSlim Lock { get; set; }

        /// <summary>
        /// Gets or sets a semaphore to synchronize visitor instances to make sure nodes aren't passed through multiple times during a pass through the computational graph.
        /// </summary>
        SemaphoreSlim? SyncSemaphore { get; set; }

        /// <summary>
        /// Gets or sets the property to store the gradient destination objects.
        /// </summary>
        object[] GradientDestinations { get; set; }

        /// <summary>
        /// Gets or sets the property to store the name of the result variable.
        /// </summary>
        string ResultToName { get; set; }

        /// <summary>
        /// Gets or sets the layer info.
        /// </summary>
        LayerInfo LayerInfo { get; set; }

        /// <summary>
        /// Initializes the operation from the given operation info.
        /// </summary>
        /// <param name="info">The operation info.</param>
        /// <param name="gradients">The gradients.</param>
        /// <param name="layerInfo" >The layer info.</param>
        void InitializeFrom(OperationInfo info, ConcurrentDictionary<string, Func<LayerInfo, object>> gradients, LayerInfo layerInfo);

        /// <summary>
        /// Gets the output of the operation.
        /// </summary>
        /// <returns>The output of the operation.</returns>
        Matrix GetOutput();

        /// <summary>
        /// Gets the deep output of the operation.
        /// </summary>
        /// <returns>The deep output of the operation.</returns>
        DeepMatrix GetDeepOutput();

        /// <summary>
        /// Resets the visitor count, accumulated gradients, among other things.
        /// </summary>
        void Reset();

        /// <summary>
        /// Stores the intermediates.
        /// </summary>
        /// <param name="id">The ID.</param>
        void Store(Guid id);

        /// <summary>
        /// Restores the intermediates.
        /// </summary>
        /// <param name="id">The ID.</param>
        void Restore(Guid id);

        /// <summary>
        /// Send the calculated gradient to the appropriate destination object.
        /// </summary>
        /// <param name="dOutput">The calculated gradients to accumulate.</param>
        /// <param name="accumulateGradientsDirectly">Whether to accumulate the gradients directly.</param>
        void AccumulateGradient(object?[] dOutput, bool accumulateGradientsDirectly);

        /// <summary>
        /// Copies the result of the operation to the specified destination.
        /// It uses the layer index to get the object to copy the result to.
        /// </summary>
        /// <param name="func">The function to get the object to copy the result to.</param>
        void ResultTo(Func<int, object> func);

        /// <summary>
        /// Copies the result of the operation to the specified destination.
        /// It uses the time step index and layer index to get the object to copy the result to.
        /// </summary>
        /// <param name="func">The function to get the object to copy the result to.</param>
        void ResultTo(Func<int, int, object> func);

        /// <summary>
        /// Copies the result of the operation to the destination.
        /// </summary>
        /// <param name="graph">The computation graph.</param>
        void ResultTo(ComputationGraph graph);

        /// <summary>
        /// Copies the result of the operation to the specified destination.
        /// </summary>
        /// <param name="objToCopy">Either a Matrix or a DeepMatrix or an Operation.</param>
        void CopyResult(object objToCopy);

        /// <summary>
        /// Replaces the result of the operation to the specified destination.
        /// If the inner arrays are different sizes, this operation will keep the differing sizes, whereas CopyResult does not handle differing sizes.
        /// </summary>
        /// <param name="objToCopy">Either a Matrix or a DeepMatrix or an Operation.</param>
        void ReplaceResult(object objToCopy);

        /// <summary>
        /// Initialize the operation with the specified starting point index.
        /// </summary>
        /// <param name="startingPointIndex">The starting point index.</param>
        void Initialize(int startingPointIndex);

        /// <summary>
        /// Initialize the lock object.
        /// </summary>
        void InitializeLock();

        /// <summary>
        /// Initialize the synchronization semaphore.
        /// </summary>
        void InitializeSyncSemaphore();
    }
}

//------------------------------------------------------------------------------
// <copyright file="IAverageableModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using Newtonsoft.Json;

    /// <summary>
    /// Am interface to a model layer.
    /// </summary>
    internal interface IAverageableModelLayer : IModelLayer
    {
        /// <summary>
        /// Gets the model elements.
        /// </summary>
        internal ConcurrentDictionary<string, (object weight, object gradient, object firstMoment, object secondMoment, int[] dimensions, InitializationType initialization)> Elements { get; }
    }
}

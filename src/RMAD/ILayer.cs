//------------------------------------------------------------------------------
// <copyright file="ILayer.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using Newtonsoft.Json;

    /// <summary>
    /// Describes a regular or nested layer.
    /// </summary>
    [JsonConverter(typeof(ILayerConverter))]
    public interface ILayer
    {
    }
}

﻿//------------------------------------------------------------------------------
// <copyright file="Layer.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    public class Layer
    {
        public List<OperationInfo> Operations { get; set; }
    }
}
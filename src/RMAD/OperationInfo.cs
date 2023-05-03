//------------------------------------------------------------------------------
// <copyright file="OperationInfo.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class OperationInfo
    {
        public string Id { get; set; }

        public string Description { get; set; }

        public string Type { get; set; }

        public string[] Inputs { get; set; }

        public string SetResultTo { get; set; }

        public string[] GradientResultTo { get; set; }
    }
}

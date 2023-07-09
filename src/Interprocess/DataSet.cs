//------------------------------------------------------------------------------
// <copyright file="DataSet.cs" author="ameritusweb" date="7/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    using System.Collections.Generic;

    /// <summary>
    /// A data set.
    /// </summary>
    public class DataSet
    {
        /// <summary>
        /// Gets or sets data set items.
        /// </summary>
        public List<DataSetItem> Items { get; set; } = new List<DataSetItem>();
    }
}

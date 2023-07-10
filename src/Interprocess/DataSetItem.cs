//------------------------------------------------------------------------------
// <copyright file="DataSetItem.cs" author="ameritusweb" date="7/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    /// <summary>
    /// A data set item.
    /// </summary>
    public class DataSetItem
    {
        /// <summary>
        /// Gets or sets the start index.
        /// </summary>
        public int StartIndex { get; set; }

        /// <summary>
        /// Gets or sets the number of rows.
        /// </summary>
        public int Rows { get; set; }

        /// <summary>
        /// Gets or sets the deserialize key.
        /// </summary>
        public (string, int) DeserializeKey { get; set; }

        /// <summary>
        /// Gets or sets the type name.
        /// </summary>
        public string TypeName { get; set; }
    }
}

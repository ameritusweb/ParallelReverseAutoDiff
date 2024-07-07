//------------------------------------------------------------------------------
// <copyright file="IPradBatchOperationArg.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Runtime.CompilerServices;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An interface for operation arguments.
    /// </summary>
    /// <typeparam name="TOperation">The operation class.</typeparam>
    /// <typeparam name="TArg">The first argument type.</typeparam>
    public interface IPradBatchOperationArg<TOperation, TArg> : IPradBatchOperationArgs<TOperation>
        where TOperation : IBatchOperation
    {
        /// <summary>
        /// An interface for constructor arguments.
        /// </summary>
        /// <typeparam name="TC1">The first constructor argument.</typeparam>
        public interface IPradOperationConstructorArgs<TC1>
        {
            /// <summary>
            /// Gets the constructor args.
            /// </summary>
            public TC1 ConstructorArgs { get; }
        }

        /// <summary>
        /// An interface for constructor arguments.
        /// </summary>
        /// <typeparam name="TC1">The first constructor argument.</typeparam>
        /// <typeparam name="TC2">The second constructor argument.</typeparam>
        public interface IPradOperationConstructorArgs<TC1, TC2>
        {
            /// <summary>
            /// Gets the constructor args.
            /// </summary>
            public (TC1, TC2) ConstructorArgs { get; }
        }

        /// <summary>
        /// Gets the args.
        /// </summary>
        public TArg Arg { get; }
    }
}

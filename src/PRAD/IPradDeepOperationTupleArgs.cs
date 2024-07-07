//------------------------------------------------------------------------------
// <copyright file="IPradDeepOperationTupleArgs.cs" author="ameritusweb" date="6/20/2024">
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
    /// <typeparam name="TTuple">The first argument type.</typeparam>
    public interface IPradDeepOperationTupleArgs<TOperation, TTuple> : IPradDeepOperationArgs<TOperation>
        where TOperation : IDeepOperation
        where TTuple : ITuple
    {
        /// <summary>
        /// An interface for constructor arguments.
        /// </summary>
        /// <typeparam name="TC1">The first constructor argument.</typeparam>
        public interface IPradOperationConstructorArg<TC1>
        {
            /// <summary>
            /// Gets the constructor args.
            /// </summary>
            public TC1 ConstructorArg { get; }
        }

        /// <summary>
        /// An interface for constructor arguments.
        /// </summary>
        /// <typeparam name="TConstructorTuple">The first constructor argument.</typeparam>
        public interface IPradOperationConstructorTupleArgs<TConstructorTuple>
            where TConstructorTuple : ITuple
        {
            /// <summary>
            /// Gets the constructor args.
            /// </summary>
            public TConstructorTuple ConstructorArgs { get; }
        }

        /// <summary>
        /// Gets the args.
        /// </summary>
        public TTuple Args { get; }
    }

    /// <summary>
    /// A PRAD Operation.
    /// </summary>
    /// <typeparam name="TOperation">The operation.</typeparam>
    public interface IPradDeepOperationArgs<TOperation>
        where TOperation : IDeepOperation
    {
    }
}

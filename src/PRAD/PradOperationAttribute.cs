//------------------------------------------------------------------------------
// <copyright file="PradOperationAttribute.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;

    /// <summary>
    /// Attribute to mark PradOp operations.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)]
    internal sealed class PradOperationAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradOperationAttribute"/> class.
        /// </summary>
        /// <param name="staticPropertyName">The static property name.</param>
        public PradOperationAttribute(string staticPropertyName)
        {
            this.StaticPropertyName = staticPropertyName;
        }

        /// <summary>
        /// Gets the static property name.
        /// </summary>
        public string StaticPropertyName { get; }

        /// <summary>
        /// Gets the static delegate.
        /// </summary>
        public Delegate StaticDelegate =>
            (typeof(PradOp).GetProperty(this.StaticPropertyName)?.GetValue(null) as Delegate) ?? throw new InvalidOperationException($"{this.StaticPropertyName} error.");

        /// <summary>
        /// Gets the delegate type.
        /// </summary>
        public Type DelegateType => this.StaticDelegate!.GetType();
    }
}

//------------------------------------------------------------------------------
// <copyright file="ConditionallyInternalUseOnlyAttribute.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Reflection;
    using System.Runtime.CompilerServices;

    /// <summary>
    /// Marks a method as for internal use only if a condition is true and from an external DLL.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class ConditionallyInternalUseOnlyAttribute : Attribute
    {
        /// <summary>
        /// Validates the condition.
        /// </summary>
        /// <param name="condition">The condition.</param>
        /// <param name="message">The message.</param>
        /// <exception cref="InvalidOperationException">The exception.</exception>
        public void Validate(bool condition, string message)
        {
            if (condition && !Assembly.GetCallingAssembly().Equals(Assembly.GetExecutingAssembly()))
            {
                throw new InvalidOperationException(message);
            }
        }
    }
}

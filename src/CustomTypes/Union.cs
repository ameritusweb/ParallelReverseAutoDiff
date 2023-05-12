//------------------------------------------------------------------------------
// <copyright file="Union.cs" author="ameritusweb" date="5/12/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.CustomTypes
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Provides a union of two types.
    /// </summary>
    /// <typeparam name="T1">The first type.</typeparam>
    /// <typeparam name="T2">The second type.</typeparam>
    public struct Union<T1, T2>
    {
        private readonly T1 left;
        private readonly T2 right;
        private readonly bool isRight;

        /// <summary>
        /// Initializes a new instance of the <see cref="Union{T1, T2}"/> struct.
        /// </summary>
        /// <param name="left">The left value.</param>
        public Union(T1 left)
            : this()
        {
            this.left = left;
            this.isRight = false;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Union{T1, T2}"/> struct.
        /// </summary>
        /// <param name="right">The right value.</param>
        public Union(T2 right)
            : this()
        {
            this.right = right;
            this.isRight = true;
        }

        /// <summary>
        /// Gets the left value.
        /// </summary>
        public T1 Left
        {
            get
            {
                if (this.isRight)
                {
                    throw new InvalidOperationException("Cannot access Left because this union holds a Right.");
                }

                return this.left;
            }
        }

        /// <summary>
        /// Gets the right value.
        /// </summary>
        public T2 Right
        {
            get
            {
                if (!this.isRight)
                {
                    throw new InvalidOperationException("Cannot access Right because this union holds a Left.");
                }

                return this.right;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this union holds a right value.
        /// </summary>
        public bool IsRight => this.isRight;

        /// <summary>
        /// Converts a left value to a union.
        /// </summary>
        /// <param name="left">The left value.</param>
        public static implicit operator Union<T1, T2>(T1 left) => new Union<T1, T2>(left);

        /// <summary>
        /// Converts a right value to a union.
        /// </summary>
        /// <param name="right">The right value.</param>
        public static implicit operator Union<T1, T2>(T2 right) => new Union<T1, T2>(right);

        /// <summary>
        /// Matches the union to either a left or right value.
        /// </summary>
        /// <param name="leftAction">The left action.</param>
        /// <param name="rightAction">The right action.</param>
        public void Match(Action<T1> leftAction, Action<T2> rightAction)
        {
            if (this.isRight)
            {
                rightAction(this.Right);
            }
            else
            {
                leftAction(this.Left);
            }
        }

        /// <summary>
        /// Overrides the equality operator.
        /// </summary>
        /// <param name="obj">The object to compare.</param>
        /// <returns>The comparison.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Union<T1, T2> other)
            {
                if (this.isRight != other.isRight)
                {
                    return false;
                }

                if (this.isRight)
                {
                    return EqualityComparer<T2>.Default.Equals(this.right, other.right);
                }

                return EqualityComparer<T1>.Default.Equals(this.left, other.left);
            }

            return false;
        }

        /// <summary>
        /// Gets the hash code.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return this.isRight ? EqualityComparer<T2>.Default.GetHashCode(this.right) : EqualityComparer<T1>.Default.GetHashCode(this.left);
        }
    }
}

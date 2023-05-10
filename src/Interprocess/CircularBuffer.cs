//------------------------------------------------------------------------------
// <copyright file="CircularBuffer.cs" author="ameritusweb" date="5/9/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    using System;
    using System.Runtime.CompilerServices;

    /// <summary>
    /// A circular buffer.
    /// </summary>
    internal sealed class CircularBuffer
    {
        private readonly object syncLock = new object(); // Object for synchronization.
        private byte[] buffer;

        /// <summary>
        /// Initializes a new instance of the <see cref="CircularBuffer"/> class.
        /// </summary>
        /// <param name="capacity">The capacity.</param>
        internal CircularBuffer(int capacity)
        {
            this.buffer = new byte[capacity];
            this.Capacity = capacity;
        }

        /// <summary>
        /// Gets the buffer's capacity.
        /// </summary>
        internal int Capacity
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
            private set;
        }

        /// <summary>
        /// Gets the pointer at the specified offset.
        /// </summary>
        /// <param name="offset">The offset.</param>
        /// <param name="length">The length.</param>
        /// <returns>The pointer.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Span<byte> GetSpan(int offset, int length)
        {
            // Lock around the operation
            lock (this.syncLock)
            {
                this.AdjustedOffset(ref offset);
                return this.buffer.AsSpan(offset, length);
            }
        }

        /// <summary>
        /// Method to read data from the buffer at the specified offset and length.
        /// </summary>
        /// <param name="offset">The offset.</param>
        /// <param name="length">The length.</param>
        /// <returns>The read memory.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal ReadOnlyMemory<byte> Read(int offset, int length)
        {
            // Lock around the operation
            lock (this.syncLock)
            {
                if (length == 0)
                {
                    return ReadOnlyMemory<byte>.Empty;
                }

                var result = new byte[length];
                var resultMemory = result.AsMemory();
                var rightLength = Math.Min(this.Capacity - offset, length);

                if (rightLength > 0)
                {
                    this.buffer.AsSpan(offset, rightLength).CopyTo(resultMemory.Span);
                }

                var leftLength = length - rightLength;

                if (leftLength > 0)
                {
                    this.buffer.AsSpan(0, leftLength).CopyTo(resultMemory.Span.Slice(rightLength, leftLength));
                }

                return resultMemory;
            }
        }

        /// <summary>
        /// Method to write a structure to the buffer at the specified offset.
        /// </summary>
        /// <param name="source">The structure.</param>
        /// <param name="offset">The offset.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Write(ReadOnlySpan<byte> source, int offset)
        {
            // Lock around the operation
            lock (this.syncLock)
            {
                var sourceLength = source.Length;
                var rightLength = Math.Min(this.Capacity - offset, sourceLength);
                source.Slice(0, rightLength).CopyTo(this.buffer.AsSpan(offset, rightLength));

                var leftLength = sourceLength - rightLength;

                if (leftLength > 0)
                {
                    source.Slice(rightLength, leftLength).CopyTo(this.buffer.AsSpan(0, leftLength));
                }
            }
        }

        /// <summary>
        /// Method to clear data in the buffer at the specified offset and length.
        /// </summary>
        /// <param name="offset">The offset.</param>
        /// <param name="length">The length.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Clear(int offset, int length)
        {
            // Lock around the operation
            lock (this.syncLock)
            {
                if (length == 0)
                {
                    return;
                }

                var rightLength = Math.Min(this.Capacity - offset, length);
                this.buffer.AsSpan(offset, rightLength).Clear();

                var leftLength = length - rightLength;

                if (leftLength > 0)
                {
                    this.buffer.AsSpan(0, leftLength).Clear();
                }
            }
        }

        /// <summary>
        /// Internal for testing.
        /// </summary>
        /// <param name="offset">The offset.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void AdjustedOffset(ref int offset)
        {
            // Lock around the operation
            lock (this.syncLock)
            {
                offset %= this.Capacity;
            }
        }

        /// <summary>
        /// Resizes the circular buffer to the specified capacity.
        /// </summary>
        /// <param name="newCapacity">The new capacity.</param>
        internal void Resize(int newCapacity)
        {
            if (newCapacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(newCapacity), "The new capacity must be greater than 0.");
            }

            // Lock around the operation
            lock (this.syncLock)
            {
                if (newCapacity == this.Capacity)
                {
                    return;
                }

                var newBuffer = new byte[newCapacity];
                var oldLength = this.buffer.Length;
                var copyLength = Math.Min(oldLength, newCapacity);

                if (copyLength > 0)
                {
                    var rightLength = Math.Min(oldLength - copyLength, copyLength);
                    if (rightLength > 0)
                    {
                        this.buffer.AsSpan(oldLength - rightLength, rightLength).CopyTo(newBuffer.AsSpan(0, rightLength));
                    }

                    var leftLength = copyLength - rightLength;
                    if (leftLength > 0)
                    {
                        this.buffer.AsSpan(0, leftLength).CopyTo(newBuffer.AsSpan(rightLength, leftLength));
                    }
                }

                this.buffer = newBuffer;
                this.Capacity = newCapacity;
            }
        }
    }
}
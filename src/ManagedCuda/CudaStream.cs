﻿// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using System;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda
{
    /// <summary>
    /// Wrapps a CUstream handle.
    /// In case of a so called NULL stream, use the native CUstream struct instead. 
    /// </summary>
    public class CudaStream : IDisposable
    {
        private bool disposed;
        private CUResult res;
        private CUstream _stream;
        private bool _isOwner;

        #region Constructor
        /// <summary>
        /// Creates a new Stream using <see cref="CUStreamFlags.None"/>
        /// </summary>
        public CudaStream()
            : this(CUStreamFlags.None)
        {
        }

        /// <summary>
        /// Creates a new wrapper for an existing stream
        /// </summary>
		public CudaStream(CUstream stream)
        {
            _stream = stream;
            _isOwner = false;
        }

        /// <summary>
        /// Creates a new Stream
        /// </summary>
        /// <param name="flags">Parameters for stream creation (must be <see cref="CUStreamFlags.None"/>)</param>
        public CudaStream(CUStreamFlags flags)
        {
            _stream = new CUstream();

            res = DriverAPINativeMethods.Streams.cuStreamCreate(ref _stream, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
		/// Creates a new Stream using <see cref="CUStreamFlags.None"/> and with the given priority<para/>
		/// This API alters the scheduler priority of work in the stream. Work in a higher priority stream 
		/// may preempt work already executing in a low priority stream.<para/>
		/// <c>priority</c> follows a convention where lower numbers represent higher priorities.<para/>
		/// '0' represents default priority.
        /// </summary>
		/// <param name="priority">Stream priority. Lower numbers represent higher priorities.</param>
        public CudaStream(int priority)
            : this(priority, CUStreamFlags.None)
        {
        }

        /// <summary>
        /// Creates a new Stream using <see cref="CUStreamFlags.None"/> and with the given priority<para/>
        /// This API alters the scheduler priority of work in the stream. Work in a higher priority stream 
        /// may preempt work already executing in a low priority stream.<para/>
        /// <c>priority</c> follows a convention where lower numbers represent higher priorities.<para/>
        /// '0' represents default priority.
        /// </summary>
        /// <param name="priority">Stream priority. Lower numbers represent higher priorities.</param>
        /// <param name="flags">Parameters for stream creation (must be <see cref="CUStreamFlags.None"/>)</param>
        public CudaStream(int priority, CUStreamFlags flags)
        {
            _stream = new CUstream();

            res = DriverAPINativeMethods.Streams.cuStreamCreateWithPriority(ref _stream, flags, priority);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamCreateWithPriority", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaStream()
        {
            Dispose(false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// For IDisposable
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed && _isOwner)
            {
                res = DriverAPINativeMethods.Streams.cuStreamDestroy_v2(_stream);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed && _isOwner)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Properties
        /// <summary>
        /// returns the wrapped CUstream handle
        /// </summary>
        public CUstream Stream
        {
            get { return _stream; }
            set { _stream = value; }
        }

        /// <summary>
        /// Returns the unique Id associated with the stream handle
        /// </summary>
        public ulong ID
        {
            get { return _stream.ID; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Waits until the device has completed all operations in the stream. If the context was created
        /// with the <see cref="CUCtxFlags.BlockingSync"/> flag, the CPU thread will block until the stream is finished with all of its
        /// tasks.
        /// </summary>
        public void Synchronize()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            res = DriverAPINativeMethods.Streams.cuStreamSynchronize(_stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamSynchronize", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Returns true if all operations in the stream have completed, or
        /// false if not.
        /// </summary>
        /// <returns></returns>
        public bool Query()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            res = DriverAPINativeMethods.Streams.cuStreamQuery(_stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamQuery", res));
            if (res != CUResult.Success && res != CUResult.ErrorNotReady) throw new CudaException(res);

            if (res == CUResult.Success) return true;
            return false; // --> ErrorNotReady
        }

        /// <summary>
        /// Make a compute stream wait on an event<para/>
        /// Makes all future work submitted to the Stream wait until <c>hEvent</c>
        /// reports completion before beginning execution. This synchronization
        /// will be performed efficiently on the device.
        /// <para/>
        /// The stream will wait only for the completion of the most recent
        /// host call to <see cref="CudaEvent.Record()"/> on <c>hEvent</c>. Once this call has returned,
        /// any functions (including <see cref="CudaEvent.Record()"/> and <see cref="Dispose()"/> may be
        /// called on <c>hEvent</c> again, and the subsequent calls will not have any
        /// effect on this stream.
        /// <para/>
        /// If <c>hStream</c> is 0 (the NULL stream) any future work submitted in any stream
        /// will wait for <c>hEvent</c> to complete before beginning execution. This
        /// effectively creates a barrier for all future work submitted to the context.
        /// <para/>
        /// If <see cref="CudaEvent.Record()"/> has not been called on <c>hEvent</c>, this call acts as if
        /// the record has already completed, and so is a functional no-op.
        /// </summary>
        /// <returns></returns>
        public void WaitEvent(CUevent cuevent)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            res = DriverAPINativeMethods.Streams.cuStreamWaitEvent(_stream, cuevent, 0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamWaitEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);

        }

        /// <summary> 
        /// Adds a callback to be called on the host after all currently enqueued
        /// items in the stream have completed.  For each 
        /// cuStreamAddCallback call, the callback will be executed exactly once.
        /// The callback will block later work in the stream until it is finished.
        /// <para/>
        /// The callback may be passed <see cref="CUResult.Success"/> or an error code.  In the event
        /// of a device error, all subsequently executed callbacks will receive an
        /// appropriate <see cref="CUResult"/>.
        /// <para/>
        /// Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
        /// will result in <see cref="CUResult.ErrorNotPermitted"/>.  Callbacks must not perform any
        /// synchronization that may depend on outstanding device work or other callbacks
        /// that are not mandated to run earlier.  Callbacks without a mandated order
        /// (in independent streams) execute in undefined order and may be serialized.
        /// <para/>
        /// This API requires compute capability 1.1 or greater.  See
        /// cuDeviceGetAttribute or ::cuDeviceGetProperties to query compute
        /// capability.  Attempting to use this API with earlier compute versions will
        /// return <see cref="CUResult.ErrorNotSupported"/>.
        /// </summary>
        /// <param name="callback">The function to call once preceding stream operations are complete</param>
        /// <param name="userData">User specified data to be passed to the callback function. Use GCAlloc to pin a managed object</param>
        /// <param name="flags">Callback flags (must be CUStreamAddCallbackFlags.None)</param>
        public void AddCallback(CUstreamCallback callback, IntPtr userData, CUStreamAddCallbackFlags flags)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            res = DriverAPINativeMethods.Streams.cuStreamAddCallback(_stream, callback, userData, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAddCallback", res));
            if (res != CUResult.Success) throw new CudaException(res);

        }

        /// <summary> 
        /// Here the Stream is the NULL stream<para/>
        /// Adds a callback to be called on the host after all currently enqueued
        /// items in the stream have completed.  For each 
        /// cuStreamAddCallback call, the callback will be executed exactly once.
        /// The callback will block later work in the stream until it is finished.
        /// <para/>
        /// The callback may be passed <see cref="CUResult.Success"/> or an error code.  In the event
        /// of a device error, all subsequently executed callbacks will receive an
        /// appropriate <see cref="CUResult"/>.
        /// <para/>
        /// Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
        /// will result in <see cref="CUResult.ErrorNotPermitted"/>.  Callbacks must not perform any
        /// synchronization that may depend on outstanding device work or other callbacks
        /// that are not mandated to run earlier.  Callbacks without a mandated order
        /// (in independent streams) execute in undefined order and may be serialized.
        /// <para/>
        /// This API requires compute capability 1.1 or greater.  See
        /// cuDeviceGetAttribute or ::cuDeviceGetProperties to query compute
        /// capability.  Attempting to use this API with earlier compute versions will
        /// return <see cref="CUResult.ErrorNotSupported"/>.
        /// </summary>
        /// <param name="callback">The function to call once preceding stream operations are complete</param>
        /// <param name="userData">User specified data to be passed to the callback function. Use GCAlloc to pin a managed object</param>
        /// <param name="flags">Callback flags (must be CUStreamAddCallbackFlags.None)</param>
        public static void AddCallbackToNullStream(CUstreamCallback callback, IntPtr userData, CUStreamAddCallbackFlags flags)
        {
            CUResult res = DriverAPINativeMethods.Streams.cuStreamAddCallback(new CUstream(), callback, userData, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAddCallback", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Query the priority of this stream
        /// </summary>
        /// <returns>the stream's priority</returns>
        public int GetPriority()
        {
            int priority = 0;
            CUResult res = DriverAPINativeMethods.Streams.cuStreamGetPriority(_stream, ref priority);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamGetPriority", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return priority;
        }

        /// <summary>
        /// Query the flags of this stream.
        /// </summary>
        /// <returns>the stream's flags<para/>
        /// The value returned in <c>flags</c> is a logical 'OR' of all flags that
        /// were used while creating this stream.</returns>
        public CUStreamFlags cuStreamGetFlags()
        {
            CUStreamFlags flags = 0;
            CUResult res = DriverAPINativeMethods.Streams.cuStreamGetFlags(_stream, ref flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamGetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return flags;
        }

        /// <summary>
        /// Wait on a memory location<para/>
        /// Enqueues a synchronization of the stream on the given memory location. Work
        /// ordered after the operation will block until the given condition on the
        /// memory is satisfied. By default, the condition is to wait for (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal.
        /// <para/>
        /// Other condition types can be specified via \p flags.
        /// <para/>
        /// If the memory was registered via ::cuMemHostRegister(), the device pointer
        /// should be obtained with::cuMemHostGetDevicePointer(). This function cannot
        /// be used with managed memory(::cuMemAllocManaged).
        /// <para/>
        /// Support for this can be queried with ::cuDeviceGetAttribute() and
        /// ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The only requirement for basic
        /// support is that on Windows, a device must be in TCC mode.
        /// </summary>
        /// <param name="addr">The memory location to wait on.</param>
        /// <param name="value">The value to compare with the memory location.</param>
        /// <param name="flags">See::CUstreamWaitValue_flags.</param>
        public void WaitValue(CUdeviceptr addr, uint value, CUstreamWaitValue_flags flags)
        {
            CUResult res = DriverAPINativeMethods.Events.cuStreamWaitValue32(_stream, addr, value, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamWaitValue32", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Wait on a memory location<para/>
        /// Enqueues a synchronization of the stream on the given memory location. Work
        /// ordered after the operation will block until the given condition on the
        /// memory is satisfied. By default, the condition is to wait for (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal.
        /// <para/>
        /// Other condition types can be specified via \p flags.
        /// <para/>
        /// If the memory was registered via ::cuMemHostRegister(), the device pointer
        /// should be obtained with::cuMemHostGetDevicePointer(). This function cannot
        /// be used with managed memory(::cuMemAllocManaged).
        /// <para/>
        /// Support for this can be queried with ::cuDeviceGetAttribute() and
        /// ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The requirements are
        /// compute capability 7.0 or greater, and on Windows, that the device be in
        /// TCC mode.
        /// </summary>
        /// <param name="addr">The memory location to wait on.</param>
        /// <param name="value">The value to compare with the memory location.</param>
        /// <param name="flags">See::CUstreamWaitValue_flags.</param>
        public void WaitValue(CUdeviceptr addr, ulong value, CUstreamWaitValue_flags flags)
        {
            CUResult res = DriverAPINativeMethods.Events.cuStreamWaitValue64(_stream, addr, value, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamWaitValue64", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Write a value to memory
        /// <para/>
        /// Write a value to memory.Unless the ::CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
        /// flag is passed, the write is preceded by a system-wide memory fence,
        /// equivalent to a __threadfence_system() but scoped to the stream
        /// rather than a CUDA thread.
        /// <para/>
        /// If the memory was registered via ::cuMemHostRegister(), the device pointer
        /// should be obtained with::cuMemHostGetDevicePointer(). This function cannot
        /// be used with managed memory(::cuMemAllocManaged).
        /// <para/>
        /// Support for this can be queried with ::cuDeviceGetAttribute() and
        /// ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The only requirement for basic
        /// support is that on Windows, a device must be in TCC mode.
        /// </summary>
        /// <param name="addr">The device address to write to.</param>
        /// <param name="value">The value to write.</param>
        /// <param name="flags">See::CUstreamWriteValue_flags.</param>
        public void WriteValue(CUdeviceptr addr, uint value, CUstreamWriteValue_flags flags)
        {
            CUResult res = DriverAPINativeMethods.Events.cuStreamWriteValue32(_stream, addr, value, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamWriteValue32", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Write a value to memory
        /// <para/>
        /// Write a value to memory.Unless the ::CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
        /// flag is passed, the write is preceded by a system-wide memory fence,
        /// equivalent to a __threadfence_system() but scoped to the stream
        /// rather than a CUDA thread.
        /// <para/>
        /// If the memory was registered via ::cuMemHostRegister(), the device pointer
        /// should be obtained with::cuMemHostGetDevicePointer(). This function cannot
        /// be used with managed memory(::cuMemAllocManaged).
        /// <para/>
        /// Support for this can be queried with ::cuDeviceGetAttribute() and
        /// ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The requirements are
        /// compute capability 7.0 or greater, and on Windows, that the device be in
        /// TCC mode.
        /// </summary>
        /// <param name="addr">The device address to write to.</param>
        /// <param name="value">The value to write.</param>
        /// <param name="flags">See::CUstreamWriteValue_flags.</param>
        public void WriteValue(CUdeviceptr addr, ulong value, CUstreamWriteValue_flags flags)
        {
            CUResult res = DriverAPINativeMethods.Events.cuStreamWriteValue64(_stream, addr, value, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamWriteValue64", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Copies attributes from source stream to destination stream<para/>
        /// Copies attributes from source stream \p src to destination stream \p dst.<para/>
        /// Both streams must have the same context.
        /// </summary>
        /// <param name="dst">Destination stream</param>
        public void CopyAttributes(CudaStream dst)
        {
            CUResult res = DriverAPINativeMethods.Streams.cuStreamCopyAttributes(dst.Stream, _stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamCopyAttributes", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Queries stream attribute.<para/>
        /// Queries attribute \p attr from \p hStream and stores it in corresponding member of \p value_out.
        /// </summary>
        /// <param name="attr"></param>
        public CUstreamAttrValue GetAttribute(CUstreamAttrID attr)
        {
            CUstreamAttrValue value = new CUstreamAttrValue();
            CUResult res = DriverAPINativeMethods.Streams.cuStreamGetAttribute(_stream, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamGetAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return value;
        }


        /// <summary>
        /// Sets stream attribute.<para/>
        /// Sets attribute \p attr on \p hStream from corresponding attribute of
        /// value.The updated attribute will be applied to subsequent work
        /// submitted to the stream. It will not affect previously submitted work.
        /// </summary>
        /// <param name="attr"></param>
        /// <param name="value"></param>
        public void SetAttribute(CUstreamAttrID attr, CUstreamAttrValue value)
        {
            CUResult res = DriverAPINativeMethods.Streams.cuStreamGetAttribute(_stream, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamGetAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
    }
}

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
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// CudaSurface3D
    /// </summary>
    [Obsolete("Texture and surface references are deprecated since CUDA 11")]
    public class CudaSurface : IDisposable
    {
        private CUsurfref _surfref;
        private CUSurfRefSetFlags _flags;
        private CUArrayFormat _format;
        private SizeT _height;
        private SizeT _width;
        private SizeT _depth;
        private uint _channelSize;
        private SizeT _dataSize;
        private int _numChannels;
        private string _name;
        private CUmodule _module;
        private CUfunction _cufunction;
        private CudaArray3D _array;
        private CUResult res;
        private bool disposed;
        private bool _isOwner;

        #region Construtors
        /// <summary>
        /// Creates a new surface from array memory. Allocates new array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels"></param>
        /// <param name="arrayFlags"></param>
        public CudaSurface(CudaKernel kernel, string surfName, CUSurfRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels, CUDAArray3DFlags arrayFlags)
        {
            _surfref = new CUsurfref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef(ref _surfref, kernel.CUModule, surfName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);

            _flags = flags;
            _format = format;
            _height = height;
            _width = width;
            _depth = depth;
            _numChannels = (int)numChannels;
            _name = surfName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * depth * _numChannels * _channelSize;
            _array = new CudaArray3D(format, width, height, depth, numChannels, arrayFlags);

            res = DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray(_surfref, _array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// Creates a new surface from array memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public CudaSurface(CudaKernel kernel, string surfName, CUSurfRefSetFlags flags, CudaArray3D array)
        {
            _surfref = new CUsurfref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef(ref _surfref, kernel.CUModule, surfName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);

            _flags = flags;
            _format = array.Array3DDescriptor.Format;
            _height = array.Height;
            _width = array.Width;
            _depth = array.Depth;
            _numChannels = (int)array.Array3DDescriptor.NumChannels;
            _name = surfName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;
            _channelSize = CudaHelperMethods.GetChannelSize(array.Array3DDescriptor.Format);
            _dataSize = array.Height * array.Width * array.Depth * array.Array3DDescriptor.NumChannels * _channelSize;
            _array = array;

            res = DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray(_surfref, _array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = false;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaSurface()
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
            if (fDisposing && !disposed)
            {
                if (_isOwner)
                {
                    //Dispose the internal array only if it was created internally
                    _array.Dispose();
                }
                disposed = true;
                // the _surfref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Properties
        /// <summary>
        /// SurfaceReference
        /// </summary>
        public CUsurfref SurfaceReference
        {
            get { return _surfref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUSurfRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Height
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
        }

        /// <summary>
        /// Width
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
        }

        /// <summary>
        /// ChannelSize
        /// </summary>
        public uint ChannelSize
        {
            get { return _channelSize; }
        }

        /// <summary>
        /// TotalSizeInBytes
        /// </summary>
        public SizeT TotalSizeInBytes
        {
            get { return _dataSize; }
        }

        /// <summary>
        /// NumChannels
        /// </summary>
        public int NumChannels
        {
            get { return _numChannels; }
        }

        /// <summary>
        /// Name
        /// </summary>
        public string Name
        {
            get { return _name; }
        }

        /// <summary>
        /// Module
        /// </summary>
        public CUmodule Module
        {
            get { return _module; }
        }

        /// <summary>
        /// CUFuntion
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Array
        /// </summary>
        public CudaArray3D Array
        {
            get { return _array; }
        }
        #endregion

        #region Static
        /// <summary>
        /// Create a new CudaArray3D and bind it to a surface reference.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels"></param>
        /// <param name="arrayFlags"></param>
        public static CudaArray3D BindArray(CudaKernel kernel, string surfName, CUSurfRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels, CUDAArray3DFlags arrayFlags)
        {
            CUsurfref surfref = new CUsurfref();
            CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef(ref surfref, kernel.CUModule, surfName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);

            CudaArray3D array = new CudaArray3D(format, width, height, depth, numChannels, arrayFlags);

            res = DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray(surfref, array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return array;
        }

        /// <summary>
        /// Bind a CudaArray3D to a surface reference.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public static void BindArray(CudaKernel kernel, string surfName, CUSurfRefSetFlags flags, CudaArray3D array)
        {
            CUsurfref surfref = new CUsurfref();
            CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef(ref surfref, kernel.CUModule, surfName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray(surfref, array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
    }
}

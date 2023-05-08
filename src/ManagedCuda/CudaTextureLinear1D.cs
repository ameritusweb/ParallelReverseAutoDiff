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
    /// CudaLinearTexture1D
    /// </summary>
	[Obsolete("Texture and surface references are deprecated since CUDA 11")]
    public class CudaTextureLinear1D<T> : IDisposable where T : struct
    {
        private CUtexref _texref;
        private CUFilterMode _filtermode;
        private CUTexRefSetFlags _flags;
        private CUAddressMode _addressMode0;
        private CUArrayFormat _format;
        private SizeT _size;
        private uint _channelSize;
        private SizeT _dataSize;
        private int _numChannels;
        private string _name;
        private CUmodule _module;
        private CUfunction _cufunction;
        private CudaDeviceVariable<T> _devVar;
        private CUResult res;
        private bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 1D texture from linear memory. Allocates a new device variable
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="size">In elements</param>
        public CudaTextureLinear1D(CudaKernel kernel, string texName, CUTexRefSetFlags flags, CUAddressMode addressMode0, CUArrayFormat format, SizeT size)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, CUFilterMode.Point); //Textures from linear memory can only by point filtered
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = CUFilterMode.Point;
            _flags = flags;
            _addressMode0 = addressMode0;
            _format = format;
            _size = size;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = _size * (SizeT)_numChannels * _channelSize;//;
            _devVar = new CudaDeviceVariable<T>(_size);

            SizeT NULL = 0;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress_v2(ref NULL, _texref, _devVar.DevicePointer, _dataSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 1D texture from linear memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="deviceVar"></param>
        public CudaTextureLinear1D(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUTexRefSetFlags flags, CUArrayFormat format, CudaDeviceVariable<T> deviceVar)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, CUFilterMode.Point);//Textures from linear memory can only by point filtered
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = CUFilterMode.Point;
            _flags = flags;
            _addressMode0 = addressMode0;
            _format = format;
            _size = deviceVar.Size;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = deviceVar.SizeInBytes;
            _devVar = deviceVar;

            SizeT NULL = 0;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress_v2(ref NULL, _texref, _devVar.DevicePointer, _dataSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaTextureLinear1D()
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
                _devVar.Dispose();
                disposed = true;
                // the _texref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Properties
        /// <summary>
        /// TextureReference
        /// </summary>
        public CUtexref TextureReference
        {
            get { return _texref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUTexRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode0
        {
            get { return _addressMode0; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Filtermode
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
        }

        /// <summary>
        /// Size
        /// </summary>
        public SizeT Size
        {
            get { return _size; }
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
        /// CUFunction
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Device variable in linear Memory
        /// </summary>
        public CudaDeviceVariable<T> DeviceVariable
        {
            get { return _devVar; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Binds a linear address range to the texture reference. <para/>
        /// Any previous address or CUDA array state associated with the texture reference is superseded by this function. <para/>
        /// Any memory previously bound to the texture reference is unbound.<para/>
        /// Size my differ to the previous bound variable, but type must be the same.
        /// </summary>
        /// <param name="deviceVar">New device variable to bind this texture reference to.</param>
        public void Reset(CudaDeviceVariable<T> deviceVar)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            _size = deviceVar.Size;
            _dataSize = deviceVar.SizeInBytes;
            _devVar = deviceVar;

            SizeT NULL = 0;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress_v2(ref NULL, _texref, _devVar.DevicePointer, _dataSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
    }

}
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
    /// CudaArrayTexture3D
    /// </summary>
	[Obsolete("Texture and surface references are deprecated since CUDA 11")]
    public class CudaTextureMipmappedArray : IDisposable
    {
        private CUtexref _texref;
        private CUFilterMode _filtermode;
        private CUTexRefSetFlags _flags;
        private CUAddressMode _addressMode0;
        private CUAddressMode _addressMode1;
        private CUAddressMode _addressMode2;
        private CUDAArray3DDescriptor _arrayDescriptor;
        private string _name;
        private CUmodule _module;
        private CUfunction _cufunction;
        private CudaMipmappedArray _array;
        private uint _maxAniso;
        private CUFilterMode _mipmapFilterMode;
        private float _mipmapLevelBias;
        private float _minMipmapLevelClamp;
        private float _maxMipmapLevelClamp;
        private CUResult res;
        private bool disposed;

        #region Constructors
        /// <summary>
		/// Creates a new mipmapped texture from array memory. Allocates a new mipmapped array. 
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="descriptor"></param>
        /// <param name="numMipmapLevels"></param>
        /// <param name="maxAniso"></param>
        /// <param name="mipmapFilterMode"></param>
        /// <param name="mipmapLevelBias"></param>
        /// <param name="minMipmapLevelClamp"></param>
        /// <param name="maxMipmapLevelClamp"></param>
        public CudaTextureMipmappedArray(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions,
            CUFilterMode filterMode, CUTexRefSetFlags flags, CUDAArray3DDescriptor descriptor, uint numMipmapLevels,
            uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, descriptor,
            numMipmapLevels, maxAniso, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp)
        {

        }

        /// <summary>
		/// Creates a new mipmapped texture from array memory. Allocates a new mipmapped array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="addressMode2"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="descriptor"></param>
        /// <param name="numMipmapLevels"></param>
        /// <param name="maxAniso"></param>
        /// <param name="mipmapFilterMode"></param>
        /// <param name="mipmapLevelBias"></param>
        /// <param name="minMipmapLevelClamp"></param>
        /// <param name="maxMipmapLevelClamp"></param>
        public CudaTextureMipmappedArray(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2,
            CUFilterMode filterMode, CUTexRefSetFlags flags, CUDAArray3DDescriptor descriptor, uint numMipmapLevels,
            uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
        {
            _maxAniso = maxAniso;
            _mipmapFilterMode = mipmapFilterMode;
            _mipmapLevelBias = mipmapLevelBias;
            _minMipmapLevelClamp = minMipmapLevelClamp;
            _maxMipmapLevelClamp = maxMipmapLevelClamp;

            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 2, addressMode2);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, descriptor.Format, (int)descriptor.NumChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _addressMode2 = addressMode2;
            _arrayDescriptor = descriptor;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _array = new CudaMipmappedArray(descriptor, numMipmapLevels);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmappedArray(_texref, _array.CUMipmappedArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmappedArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMaxAnisotropy(_texref, maxAniso);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMaxAnisotropy", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapFilterMode(_texref, mipmapFilterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelBias(_texref, mipmapLevelBias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelBias", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelClamp(_texref, minMipmapLevelClamp, maxMipmapLevelClamp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelClamp", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
		/// Creates a new mipmapped texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        /// <param name="maxAniso"></param>
        /// <param name="mipmapFilterMode"></param>
        /// <param name="mipmapLevelBias"></param>
        /// <param name="minMipmapLevelClamp"></param>
        /// <param name="maxMipmapLevelClamp"></param>
		public CudaTextureMipmappedArray(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaMipmappedArray array,
            uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, array,
            maxAniso, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp)
        {

        }

        /// <summary>
		/// Creates a new mipmapped texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="addressMode2"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        /// <param name="maxAniso"></param>
        /// <param name="mipmapFilterMode"></param>
        /// <param name="mipmapLevelBias"></param>
        /// <param name="minMipmapLevelClamp"></param>
        /// <param name="maxMipmapLevelClamp"></param>
		public CudaTextureMipmappedArray(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2,
            CUFilterMode filterMode, CUTexRefSetFlags flags, CudaMipmappedArray array,
            uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
        {
            _maxAniso = maxAniso;
            _mipmapFilterMode = mipmapFilterMode;
            _mipmapLevelBias = mipmapLevelBias;
            _minMipmapLevelClamp = minMipmapLevelClamp;
            _maxMipmapLevelClamp = maxMipmapLevelClamp;

            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 2, addressMode2);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, array.Array3DDescriptor.Format, (int)array.Array3DDescriptor.NumChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _addressMode2 = addressMode2;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _array = array;

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmappedArray(_texref, _array.CUMipmappedArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmappedArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMaxAnisotropy(_texref, maxAniso);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMaxAnisotropy", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapFilterMode(_texref, mipmapFilterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelBias(_texref, mipmapLevelBias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelBias", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelClamp(_texref, minMipmapLevelClamp, maxMipmapLevelClamp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelClamp", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaTextureMipmappedArray()
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
                _array.Dispose();
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
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode1
        {
            get { return _addressMode1; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode2
        {
            get { return _addressMode2; }
        }


        /// <summary>
        /// Filtermode
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
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
        public CudaMipmappedArray Array
        {
            get { return _array; }
        }

        /// <summary>
        /// MaxAniso
        /// </summary>
        public uint MaxAniso
        {
            get { return _maxAniso; }
        }

        /// <summary>
        /// Mipmap Filtermode
        /// </summary>
        public CUFilterMode MipmapFiltermode
        {
            get { return _mipmapFilterMode; }
        }

        /// <summary>
        /// MipmapLevelBias
        /// </summary>
        public float MipmapLevelBias
        {
            get { return _mipmapLevelBias; }
        }

        /// <summary>
        /// MinMipmapLevelClamp
        /// </summary>
        public float MinMipmapLevelClamp
        {
            get { return _minMipmapLevelClamp; }
        }

        /// <summary>
        /// MaxMipmapLevelClamp
        /// </summary>
        public float MaxMipmapLevelClamp
        {
            get { return _maxMipmapLevelClamp; }
        }
        #endregion
    }
}

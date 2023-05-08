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
    /// Wrapper for a CUgraphicsResource (directX)
    /// </summary>
    public class CudaDirectXInteropResource : ICudaGraphicsInteropResource
    {
        private CUgraphicsResource _cudaResource;
        private IntPtr _d3dResource;
        private CUGraphicsRegisterFlags _registerFlags;
        private CudaContext.DirectXVersion _version;
        private CUResult res;
        private bool disposed;
        private bool _IsRegistered;
        private bool _IsMapped;

        #region Constructors
        /// <summary>
        /// Registers a new graphics interop resource for interop with DirectX
        /// </summary>
        /// <param name="d3dResource">DirectX resource to register</param>
        /// <param name="flags">register Flags</param>
        /// <param name="version">DirectX version</param>
        public CudaDirectXInteropResource(IntPtr d3dResource, CUGraphicsRegisterFlags flags, CudaContext.DirectXVersion version)
        {
            _cudaResource = new CUgraphicsResource();
            _d3dResource = d3dResource;
            _registerFlags = flags;
            _version = version;

            switch (version)
            {
                case CudaContext.DirectXVersion.D3D9:
                    res = DirectX9NativeMethods.CUDA3.cuGraphicsD3D9RegisterResource(ref _cudaResource, _d3dResource, flags);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsD3D9RegisterResource", res));
                    break;
                case CudaContext.DirectXVersion.D3D10:
                    res = DirectX10NativeMethods.CUDA3.cuGraphicsD3D10RegisterResource(ref _cudaResource, _d3dResource, flags);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsD3D10RegisterResource", res));
                    break;
                case CudaContext.DirectXVersion.D3D11:
                    res = DirectX11NativeMethods.cuGraphicsD3D11RegisterResource(ref _cudaResource, _d3dResource, flags);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsD3D11RegisterResource", res));
                    break;
                default:
                    throw new ArgumentException("DirectX version not supported.", "dXVersion");
            }

            if (res != CUResult.Success) throw new CudaException(res);
            _IsRegistered = true;
            _IsMapped = false;
        }

        /// <summary>
        /// Registers a new graphics interop resource for interop with DirectX
        /// </summary>
        /// <param name="d3dResource">DirectX resource to register</param>
        /// <param name="flags">register Flags</param>
        /// <param name="version">DirectX version</param>
        /// <param name="mapFlags">resource mapping flags</param>
        public CudaDirectXInteropResource(IntPtr d3dResource, CUGraphicsRegisterFlags flags, CudaContext.DirectXVersion version, CUGraphicsMapResourceFlags mapFlags)
            : this(d3dResource, flags, version)
        {
            SetMapFlags(mapFlags);
        }


        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaDirectXInteropResource()
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
                if (_IsMapped)
                {
                    res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnmapResources(1, ref _cudaResource, new CUstream());
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsUnmapResources", res));
                }
                if (_IsRegistered)
                {
                    //Ignore if failing
                    res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnregisterResource(_cudaResource);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsUnregisterResource", res));
                }
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Maps the graphics resource for access by CUDA.<para/>
        /// The resource may be accessed by CUDA until it is unmapped. The graphics API from which the resource
        /// was registered should not access any resources while they are mapped by CUDA. If an application does
        /// so, the results are undefined.<para/>
        /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="Map()"/>
        /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
        /// If the resource is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown.
        /// </summary>
        /// <param name="stream"></param>
        public void Map(CUstream stream)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsMapResources(1, ref _cudaResource, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsMapResources", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _IsMapped = true;
        }

        /// <summary>
        /// Maps the graphics resource for access by CUDA.<para/>
        /// The resource may be accessed by CUDA until it is unmapped. The graphics API from which the resource
        /// was registered should not access any resources while they are mapped by CUDA. If an application does
        /// so, the results are undefined.<para/>
        /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="Map()"/>
        /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
        /// If the resource is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown.
        /// </summary>
        public void Map()
        {
            Map(new CUstream());
        }

        /// <summary>
        /// Unmaps the graphics resource.<para/>
        /// Once unmapped, the resource may not be accessed by CUDA until they are mapped again.<para/>
        /// This function provides the synchronization guarantee that any CUDA work issued in <c>stream</c> before <see cref="UnMap()"/>
        /// will complete before any subsequently issued graphics work begins.<para/>
        /// If the resource is not presently mapped for access by CUDA then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        /// <param name="stream"></param>
        public void UnMap(CUstream stream)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnmapResources(1, ref _cudaResource, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsUnmapResources", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _IsMapped = false;
        }

        /// <summary>
        /// Unmaps the graphics resource.<para/>
        /// Once unmapped, the resources in <c>resources</c> may not be accessed by CUDA until they are mapped again.<para/>
        /// This function provides the synchronization guarantee that any CUDA work issued in <c>stream</c> before <see cref="UnMap()"/>
        /// will complete before any subsequently issued graphics work begins.<para/>
        /// If the resource is not presently mapped for access by CUDA then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        public void UnMap()
        {
            UnMap(new CUstream());
        }

        /// <summary>
        /// Set <c>flags</c> for mapping the graphics resource. <para/>
        /// Changes to <c>flags</c> will take effect the next time <c>resource</c> is mapped. See <see cref="CUGraphicsMapResourceFlags"/>. <para/>
        /// If <c>resource</c> is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown. 
        /// </summary>
        /// <param name="flags"></param>
        public void SetMapFlags(CUGraphicsMapResourceFlags flags)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceSetMapFlags(_cudaResource, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceSetMapFlags", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Unregisters the wrapped resource. Better use Dispose(), as the wrapper of the unregistered resource is of no use after unregistering.
        /// </summary>
        public void Unregister()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            if (_IsMapped)
                UnMap();

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnregisterResource(_cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsUnregisterResource", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            _IsRegistered = false;
        }

        /// <summary>
        /// Returns device variable through which the mapped graphics resource may be accessed. <para/>
        /// The pointer value in the device variable may change every time that the resource is mapped.<para/>
        /// If the resource is not a buffer then it cannot be accessed via a pointer and <see cref="CUResult.ErrorNotMappedAsPointer"/>
        /// exception is thrown. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        /// <returns></returns>
        public CudaDeviceVariable<T> GetMappedPointer<T>() where T : struct
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUdeviceptr devPtr = new CUdeviceptr();
            SizeT size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceGetMappedPointer_v2(ref devPtr, ref size, _cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceGetMappedPointer", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            CudaDeviceVariable<T> retVar = new CudaDeviceVariable<T>(devPtr, false, size);

            return retVar;
        }

        /// <summary>
        /// Returns in <c>devicePtr</c> a pointer through which the mapped graphics resource may be accessed. Returns
        /// in <c>size</c> the size of the memory in bytes which may be accessed from that pointer. The value set in <c>devicePtr</c> may
        /// change every time that the resource is mapped.<para/>
        /// If the resource is not a buffer then it cannot be accessed via a pointer and <see cref="CUResult.ErrorNotMappedAsPointer"/>
        /// exception is thrown. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        /// <returns></returns>
        public void GetMappedPointer(out CUdeviceptr devicePtr, out SizeT size)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            devicePtr = new CUdeviceptr();
            size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceGetMappedPointer_v2(ref devicePtr, ref size, _cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceGetMappedPointer", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Returns a <c>CUdeviceptr</c>, a device pointer through which the mapped graphics resource may be accessed. 
        /// The value set in <c>devicePtr</c> may
        /// change every time that the resource is mapped.<para/>
        /// If the resource is not a buffer then it cannot be accessed via a pointer and <see cref="CUResult.ErrorNotMappedAsPointer"/>
        /// exception is thrown. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        /// <returns></returns>
        public CUdeviceptr GetMappedPointer()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUdeviceptr devicePtr = new CUdeviceptr();
            SizeT size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceGetMappedPointer_v2(ref devicePtr, ref size, _cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceGetMappedPointer", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return devicePtr;
        }

        /// <summary>
        /// Returns a CudaArray1D through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CudaArray1D</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        public CudaArray1D GetMappedArray1D(uint arrayIndex, uint mipLevel)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUarray array = new CUarray();
            SizeT size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsSubResourceGetMappedArray(ref array, _cudaResource, arrayIndex, mipLevel);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsSubResourceGetMappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            CudaArray1D retVar = new CudaArray1D(array, false);

            return retVar;
        }

        /// <summary>
        /// Returns a CudaArray2D through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CudaArray2D</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        public CudaArray2D GetMappedArray2D(uint arrayIndex, uint mipLevel)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUarray array = new CUarray();
            SizeT size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsSubResourceGetMappedArray(ref array, _cudaResource, arrayIndex, mipLevel);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsSubResourceGetMappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            CudaArray2D retVar = new CudaArray2D(array, false);

            return retVar;
        }

        /// <summary>
        /// Returns a CudaArray3D through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CudaArray3D</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        public CudaArray3D GetMappedArray3D(uint arrayIndex, uint mipLevel)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUarray array = new CUarray();
            SizeT size = 0;

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsSubResourceGetMappedArray(ref array, _cudaResource, arrayIndex, mipLevel);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsSubResourceGetMappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            CudaArray3D retVar = new CudaArray3D(array, false);

            return retVar;
        }

        /// <summary>
        /// Returns a CudaMipmappedArray through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CudaMipmappedArray</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <returns></returns>
        public CudaMipmappedArray GetMappedMipmappedArray(CUArrayFormat format, CudaMipmappedArrayNumChannels numChannels)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUmipmappedArray array = new CUmipmappedArray();

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceGetMappedMipmappedArray(ref array, _cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceGetMappedMipmappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            CudaMipmappedArray retVar = new CudaMipmappedArray(array, format, numChannels);

            return retVar;
        }

        /// <summary>
        /// Returns a CUarray handle through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CUarray</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        public CUarray GetMappedCUArray(uint arrayIndex, uint mipLevel)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUarray array = new CUarray();

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsSubResourceGetMappedArray(ref array, _cudaResource, arrayIndex, mipLevel);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsSubResourceGetMappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return array;
        }

        /// <summary>
        /// Returns a CUmipmappedArray handle through which the subresource of the mapped graphics resource resource which
        /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The pointer value in <c>CUmipmappedArray</c>
        /// may change every time that <c>resource</c> is mapped.<para/>
        /// If the resource is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
        /// exception is thrwon. If <c>arrayIndex</c> is not a valid array index for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If <c>mipLevel</c> is not a valid mipmap level for the resource then <see cref="CUResult.ErrorInvalidValue"/>
        /// exception is thrwon. If the resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> exception is thrwon.
        /// </summary>
        /// <returns></returns>
        public CUmipmappedArray GetMappedCUMipmappedArray()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            CUmipmappedArray array = new CUmipmappedArray();

            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsResourceGetMappedMipmappedArray(ref array, _cudaResource);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsResourceGetMappedMipmappedArray", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return array;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CUgraphicsResource GetCUgraphicsResource()
        {
            return _cudaResource;
        }

        /// <summary>
        /// 
        /// </summary>
        public void SetIsMapped()
        {
            _IsMapped = true;
        }

        /// <summary>
        /// 
        /// </summary>
        public void SetIsUnmapped()
        {
            _IsMapped = false;
        }
        #endregion

        #region Properties
        /// <summary>
        /// 
        /// </summary>
        public bool IsMapped
        {
            get { return _IsMapped; }
            internal set { _IsMapped = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool IsRegistered
        {
            get { return _IsRegistered; }
        }

        /// <summary>
        /// 
        /// </summary>
        public CUgraphicsResource CUgraphicsResource
        {
            get { return _cudaResource; }
        }
        #endregion
    }
}

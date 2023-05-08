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
using System.Runtime.Serialization;

namespace ManagedCuda.CudaBlas
{
    /// <summary>
    /// An CudaBlasException is thrown, if any wrapped call to the CUBLAS-library does not return <see cref="CublasStatus.Success"/>.
    /// </summary>
    public class CudaBlasException : Exception, ISerializable
    {

        private CublasStatus _cudaBlasError;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public CudaBlasException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected CudaBlasException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public CudaBlasException(CublasStatus error)
            : base(GetErrorMessageFromCUResult(error))
        {
            _cudaBlasError = error;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public CudaBlasException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaBlasException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaBlasException(CublasStatus error, string message, Exception exception)
            : base(message, exception)
        {
            _cudaBlasError = error;
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return _cudaBlasError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("CudaBlasError", _cudaBlasError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(CublasStatus error)
        {
            string message = string.Empty;

            switch (error)
            {
                case CublasStatus.Success:
                    message = "Any CUBLAS operation is successful.";
                    break;
                case CublasStatus.NotInitialized:
                    message = "The CUBLAS library was not initialized.";
                    break;
                case CublasStatus.AllocFailed:
                    message = "Resource allocation failed.";
                    break;
                case CublasStatus.InvalidValue:
                    message = "An invalid numerical value was used as an argument.";
                    break;
                case CublasStatus.ArchMismatch:
                    message = "An absent device architectural feature is required.";
                    break;
                case CublasStatus.MappingError:
                    message = "An access to GPU memory space failed.";
                    break;
                case CublasStatus.ExecutionFailed:
                    message = "An access to GPU memory space failed.";
                    break;
                case CublasStatus.InternalError:
                    message = "An internal operation failed.";
                    break;
                case CublasStatus.NotSupported:
                    message = "Error: Not supported.";
                    break;
                default:
                    break;
            }


            return error.ToString() + ": " + message;
        }
        #endregion

        #region Properties
        /// <summary>
        /// 
        /// </summary>
        public CublasStatus CudaBlasError
        {
            get
            {
                return _cudaBlasError;
            }
            set
            {
                _cudaBlasError = value;
            }
        }
        #endregion
    }
}

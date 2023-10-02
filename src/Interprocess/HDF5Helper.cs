//------------------------------------------------------------------------------
// <copyright file="HDF5Helper.cs" author="ameritusweb" date="7/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using HDF.PInvoke;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An HDF5 helper.
    /// </summary>
    public static class HDF5Helper
    {
        /// <summary>
        /// Serialize a matrix to a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <param name="matrix">The matrix.</param>
        public static void Serialize(FileInfo fileInfo, Matrix matrix)
        {
            // Open HDF5 file
            long fileId = H5F.create(fileInfo.FullName, H5F.ACC_TRUNC);

            for (int i = 0; i < matrix.Rows; i++)
            {
                // Create a dataspace for each row.
                long dataspaceId = H5S.create_simple(1, new[] { (ulong)matrix[i].Length }, null);

                // Create a dataset for each row.
                long datasetId = H5D.create(fileId, $"row{i}", H5T.NATIVE_DOUBLE, dataspaceId);

                // Pin the double[] in memory and get its address.
                GCHandle hnd = GCHandle.Alloc(matrix[i], GCHandleType.Pinned);
                IntPtr ptr = hnd.AddrOfPinnedObject();

                // Write the data to the dataset.
                H5D.write(datasetId, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, ptr);

                // Free the GCHandle when done
                hnd.Free();

                // Close the dataset and dataspace.
                H5D.close(datasetId);
                H5S.close(dataspaceId);
            }

            // Close the HDF5 file.
            H5F.close(fileId);
        }

        /// <summary>
        /// Deserialize a matrix from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <returns>The matrix.</returns>
        public static List<double[]> Deserialize(FileInfo fileInfo)
        {
            // Open the HDF5 file.
            long fileId = H5F.open(fileInfo.FullName, H5F.ACC_RDONLY, H5P.DEFAULT);

            // Get the group info.
            H5G.info_t groupInfo = default(H5G.info_t);
            _ = H5G.get_info(fileId, ginfo: ref groupInfo);

            // Get the number of datasets in the file.
            ulong numDatasets = groupInfo.nlinks;

            var data = new List<double[]>();

            // Loop over the datasets.
            for (int i = 0; i < (int)numDatasets; i++)
            {
                // Open each dataset.
                long datasetId = H5D.open(fileId, $"row{i}");

                // Get the dataspace and dimensions of the dataset.
                long dataspaceId = H5D.get_space(datasetId);
                ulong[] dimensions = new ulong[1];
                H5S.get_simple_extent_dims(dataspaceId, dimensions, null);

                // Create a buffer to hold the data.
                double[] buffer = new double[dimensions[0]];

                GCHandle hnd = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                IntPtr ptr = hnd.AddrOfPinnedObject();

                // Read the data into the buffer.
                H5D.read(datasetId, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, ptr);

                // Add the data to the list.
                data.Add(buffer);

                // Free the GCHandle when done
                hnd.Free();

                // Close the dataset and dataspace.
                H5D.close(datasetId);
                H5S.close(dataspaceId);
            }

            // Close the HDF5 file.
            H5F.close(fileId);

            return data;
        }

        /// <summary>
        /// Validate a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>The validation result.</returns>
        public static bool Validate(FileInfo fileInfo, List<int> sizes)
        {
            bool isValid = true;

            // Open the HDF5 file.
            long fileId = H5F.open(fileInfo.FullName, H5F.ACC_RDONLY, H5P.DEFAULT);

            // Get the group info.
            H5G.info_t groupInfo = default(H5G.info_t);
            _ = H5G.get_info(fileId, ginfo: ref groupInfo);

            // Get the number of datasets in the file.
            ulong numDatasets = groupInfo.nlinks;

            if (numDatasets < (ulong)sizes.Count)
            {
                isValid = false;
            }

            // Loop over the datasets.
            for (int i = 0; i < sizes.Count; i++)
            {
                // Open each dataset.
                long datasetId = H5D.open(fileId, $"row{i}");

                // Get the dataspace and dimensions of the dataset.
                long dataspaceId = H5D.get_space(datasetId);
                ulong[] dimensions = new ulong[1];
                H5S.get_simple_extent_dims(dataspaceId, dimensions, null);

                // Check if the row size is what you expect it to be
                if (dimensions[0] != (ulong)sizes[i])
                {
                    isValid = false;

                    // Close the dataset and dataspace.
                    H5D.close(datasetId);
                    H5S.close(dataspaceId);
                    break;
                }

                // Close the dataset and dataspace.
                H5D.close(datasetId);
                H5S.close(dataspaceId);
            }

            // Close the HDF5 file.
            H5F.close(fileId);

            return isValid;
        }

        /// <summary>
        /// Deserialize a dictionary from a file.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="fileInfo">The file info.</param>
        /// <param name="dictionary">The dictionary.</param>
        /// <param name="gets">The gets.</param>
        public static void Deserialize<T>(FileInfo fileInfo, IDictionary<string, T> dictionary, Func<T, object>[] gets)
        {
            // Deserialize metadata
            DataSet dataSet;
            using (StreamReader fileStream = File.OpenText(fileInfo.FullName + ".json"))
            using (JsonTextReader reader = new JsonTextReader(fileStream))
            {
                JsonSerializer serializer = new JsonSerializer();
                dataSet = serializer.Deserialize<DataSet>(reader) ?? throw new InvalidOperationException("Could not deserialize metadata.");
            }

            // Deserialize raw data
            var matrices = Deserialize(new FileInfo(fileInfo.FullName + ".hdf5"));

            // Now, build the dictionary using the metadata
            foreach (var item in dataSet.Items)
            {
                var key = item.DeserializeKey;
                var deserializeType = key.Item1;
                var deserializeIndex = key.Item2;

                // Grab the matrices that belong to this item
                var itemMatrices = matrices.Skip(item.StartIndex).Take(item.Rows).ToList();

                if (item.TypeName == typeof(Matrix).FullName)
                {
                    var matrix = (Matrix)gets[deserializeIndex](dictionary[deserializeType]);
                    matrix.Replace(itemMatrices.ToArray());
                }
                else if (item.TypeName == typeof(DeepMatrix).FullName)
                {
                    var deepMatrix = (DeepMatrix)gets[deserializeIndex](dictionary[deserializeType]);
                    deepMatrix.Replace(itemMatrices);
                }
                else if (item.TypeName == typeof(DeepMatrix[]).FullName)
                {
                    var deepMatrixArray = (DeepMatrix[])gets[deserializeIndex](dictionary[deserializeType]);
                    for (int i = 0; i < deepMatrixArray.Length; ++i)
                    {
                        var totalRows = deepMatrixArray[i].TotalRows;
                        deepMatrixArray[i].Replace(itemMatrices.Skip(i * totalRows).Take(totalRows).ToList());
                    }
                }
                else
                {
                    throw new InvalidOperationException($"Unknown type name: {item.TypeName}");
                }
            }
        }

        /// <summary>
        /// Serialize a list of matrices to a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <param name="matrices">The matrices.</param>
        public static void Serialize(FileInfo fileInfo, List<Matrix> matrices)
        {
            // Open HDF5 file
            long fileId = H5F.create(fileInfo.FullName, H5F.ACC_TRUNC);

            int n = 0;
            for (int m = 0; m < matrices.Count; m++)
            {
                var matrix = matrices[m];
                for (int i = 0; i < matrix.Rows; i++)
                {
                    // Create a dataspace for each row.
                    long dataspaceId = H5S.create_simple(1, new ulong[] { (ulong)matrix[i].Length }, null);

                    // Create a dataset for each row.
                    long datasetId = H5D.create(fileId, $"row{n}", H5T.NATIVE_DOUBLE, dataspaceId);

                    // Pin the double[] in memory and get its address.
                    GCHandle hnd = GCHandle.Alloc(matrix[i], GCHandleType.Pinned);
                    IntPtr ptr = hnd.AddrOfPinnedObject();

                    // Write the data to the dataset.
                    H5D.write(datasetId, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, ptr);

                    // Free the GCHandle when done
                    hnd.Free();

                    // Close the dataset and dataspace.
                    H5D.close(datasetId);
                    H5S.close(dataspaceId);
                    n++;
                }
            }

            // Close the HDF5 file.
            H5F.close(fileId);
        }

        /// <summary>
        /// Serialize a dictionary of matrices to a file.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="fileInfo">The file info.</param>
        /// <param name="elements">The elements dictionary.</param>
        /// <param name="gets">The array of get functions.</param>
        /// <returns>True if the serialization was successful.</returns>
        public static bool Serialize<T>(FileInfo fileInfo, IDictionary<string, T> elements, Func<T, object>[] gets)
        {
            List<Matrix> matrices = new List<Matrix>();
            DataSet dataSet = new DataSet();
            List<DataSetItem> items = new List<DataSetItem>();
            int index = 0;
            foreach (var pair in elements)
            {
                var key = pair.Key;

                for (int i = 0; i < gets.Length; ++i)
                {
                    var gitem = gets[i](pair.Value);
                    var startIndex = index;

                    var (matricesItem, typeName) = ParseObject(gitem);
                    matrices.AddRange(matricesItem);
                    var rows = matricesItem.Sum(x => x.Rows);
                    DataSetItem item = new DataSetItem()
                    {
                        DeserializeKey = (key, i),
                        StartIndex = startIndex,
                        Rows = rows,
                        TypeName = typeName,
                    };
                    items.Add(item);
                    index += rows;
                }
            }

            dataSet.Items.AddRange(items);

            using (StreamWriter fileStream = File.CreateText(fileInfo.FullName + ".json"))
            using (JsonTextWriter writer = new JsonTextWriter(fileStream))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Formatting = Formatting.Indented;
                serializer.Serialize(writer, dataSet);
            }

            var binaryFile = new FileInfo(fileInfo.FullName + ".hdf5");
            Serialize(binaryFile, matrices);

            return Validate(binaryFile, matrices.SelectMany(matrix => matrix.Select(row => row.Length)).ToList());
        }

        /// <summary>
        /// Parse a matrix, deep matrix, or deep matrix array from an object.
        /// </summary>
        /// <param name="o">The object.</param>
        /// <returns>The matrices.</returns>
        public static (List<Matrix>, string TypeName) ParseObject(object o)
        {
            if (o is Matrix matrix)
            {
                return (new List<Matrix>() { matrix }, typeof(Matrix).FullName);
            }
            else if (o is DeepMatrix deepMatrix)
            {
                return (deepMatrix.ToArray().ToList(), typeof(DeepMatrix).FullName);
            }
            else if (o is DeepMatrix[] deepMatrixArray)
            {
                return (deepMatrixArray.SelectMany(x => x.ToArray()).ToList(), typeof(DeepMatrix[]).FullName);
            }
            else
            {
                throw new ArgumentException("Object must be a Matrix, DeepMatrix, or DeepMatrix[].");
            }
        }
    }
}

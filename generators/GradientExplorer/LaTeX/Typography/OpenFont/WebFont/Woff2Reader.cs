//MIT, 2019-present, WinterDev 
using System.IO;
using System.Collections.Generic;
using Typography.OpenFont;
using Typography.OpenFont.IO;
using Typography.OpenFont.Tables;

//see https://www.w3.org/TR/WOFF2/

namespace Typography.WebFont
{
    class Woff2Header
    {
        //WOFF2 Header
        //UInt32 signature   0x774F4632 'wOF2'
        //UInt32 flavor  The "sfnt version" of the input font.
        //UInt32 length  Total size of the WOFF file.
        //UInt16 numTables   Number of entries in directory of font tables.
        //UInt16 reserved    Reserved; set to 0.
        //UInt32 totalSfntSize   Total size needed for the uncompressed font data, including the sfnt header,
        //directory, and font tables(including padding).
        //UInt32  totalCompressedSize Total length of the compressed data block.
        //UInt16  majorVersion    Major version of the WOFF file.
        //UInt16  minorVersion    Minor version of the WOFF file.
        //UInt32  metaOffset  Offset to metadata block, from beginning of WOFF file.
        //UInt32  metaLength  Length of compressed metadata block.
        //UInt32  metaOrigLength  Uncompressed size of metadata block.
        //UInt32  privOffset  Offset to private data block, from beginning of WOFF file.
        //UInt32  privLength Length of private data block.

        public uint flavor;
        public uint length;
        public uint numTables;
        public uint totalSfntSize;
        public uint totalCompressSize; //***
        public ushort majorVersion;
        public ushort minorVersion;
        public uint metaOffset;
        public uint metaLength;
        public uint metaOriginalLength;
        public uint privOffset;
        public uint privLength;
    }
    class Woff2TableDirectory
    {
        //TableDirectoryEntry
        //UInt8         flags           table type and flags
        //UInt32        tag	            4-byte tag(optional)
        //UIntBase128   origLength      length of original table
        //UIntBase128   transformLength transformed length(if applicable)

        public uint origLength;
        public uint transformLength;

        public Woff2TableDirectory(string name)
        {
            Name = name;
        }

        //translated values 
        public string Name { get; } //translate from tag
        public byte PreprocessingTransformation { get; set; }
        public long ExpectedStartAt { get; set; }
#if DEBUG
        public override string ToString()
        {
            return Name + " " + PreprocessingTransformation;
        }
#endif
    }


    public delegate bool BrotliDecompressStreamFunc(byte[] compressedInput, Stream decompressStream);

    public static class Woff2DefaultBrotliDecompressFunc
    {
        public static BrotliDecompressStreamFunc? DecompressHandler;
    }

    class TransformedGlyf : UnreadTableEntry
    {

        static TripleEncodingTable s_encTable = TripleEncodingTable.GetEncTable();

        public TransformedGlyf(TableHeader header, Woff2TableDirectory tableDir) : base(header)
        {
            HasCustomContentReader = true;
            TableDir = tableDir;
        }
        public Woff2TableDirectory TableDir { get; }

        public override T CreateTableEntry<T>(BinaryReader reader, OpenFontReader.TableReader<T> expectedResult)
        {
            return ReconstructGlyfTable(Header, reader, TableDir) is T t ? t : throw new System.NotSupportedException();
        }


        struct TempGlyph
        {
            public readonly ushort glyphIndex;
            public readonly short numContour;

            public ushort instructionLen;
            public bool compositeHasInstructions;
            public TempGlyph(ushort glyphIndex, short contourCount)
            {
                this.glyphIndex = glyphIndex;
                this.numContour = contourCount;

                instructionLen = 0;
                compositeHasInstructions = false;
            }
#if DEBUG
            public override string ToString()
            {
                return glyphIndex + " " + numContour;
            }
#endif
        }


        static Glyf ReconstructGlyfTable(TableHeader header, BinaryReader reader, Woff2TableDirectory woff2TableDir)
        {            
            reader.BaseStream.Position = woff2TableDir.ExpectedStartAt;
            ushort numGlyphs = reader.ReadUInt16();

            uint nContourStreamSize = reader.ReadUInt32(); //in bytes
            uint nPointsStreamSize = reader.ReadUInt32(); //in bytes
            uint flagStreamSize = reader.ReadUInt32(); //in bytes
            uint glyphStreamSize = reader.ReadUInt32(); //in bytes
            uint compositeStreamSize = reader.ReadUInt32(); //in bytes
            uint bboxStreamSize = reader.ReadUInt32(); //in bytes
            uint instructionStreamSize = reader.ReadUInt32(); //in bytes


            long expected_nCountStartAt = reader.BaseStream.Position;
            long expected_nPointStartAt = expected_nCountStartAt + nContourStreamSize;
            long expected_FlagStreamStartAt = expected_nPointStartAt + nPointsStreamSize;
            long expected_GlyphStreamStartAt = expected_FlagStreamStartAt + flagStreamSize;
            long expected_CompositeStreamStartAt = expected_GlyphStreamStartAt + glyphStreamSize;

            long expected_BboxStreamStartAt = expected_CompositeStreamStartAt + compositeStreamSize;
            long expected_InstructionStreamStartAt = expected_BboxStreamStartAt + bboxStreamSize;
            long expected_EndAt = expected_InstructionStreamStartAt + instructionStreamSize;

            //--------------------------------------------- 
            Glyph?[] readingGlyphs = new Glyph?[numGlyphs];
            TempGlyph[] allGlyphs = new TempGlyph[numGlyphs];
            List<ushort> compositeGlyphs = new List<ushort>();
            int contourCount = 0;
            for (ushort i = 0; i < numGlyphs; ++i)
            {
                short numContour = reader.ReadInt16();
                allGlyphs[i] = new TempGlyph(i, numContour);
                if (numContour > 0)
                {
                    contourCount += numContour;
                    //>0 => simple glyph
                    //-1 = compound
                    //0 = empty glyph
                }
                else if (numContour < 0)
                {
                    //composite glyph, resolve later
                    compositeGlyphs.Add(i);
                }
                else
                {

                }
            }
#if DEBUG
            if (reader.BaseStream.Position != expected_nPointStartAt)
            {
                System.Diagnostics.Debug.WriteLine("ERR!!");
            }
#endif
            var emptyGlyph = Glyf.GenerateTypefaceSpecificEmptyGlyph();
            //
            //1) nPoints stream,  npoint for each contour

            ushort[] pntPerContours = new ushort[contourCount];
            for (int i = 0; i < contourCount; ++i)
            {
                // Each of these is the number of points of that contour.
                pntPerContours[i] = Woff2Utils.Read255UInt16(reader);
            }
#if DEBUG
            if (reader.BaseStream.Position != expected_FlagStreamStartAt)
            {
                System.Diagnostics.Debug.WriteLine("ERR!!");
            }
#endif
            //2) flagStream, flags value for each point
            //each byte in flags stream represents one point
            byte[] flagStream = reader.ReadBytes((int)flagStreamSize);

#if DEBUG
            if (reader.BaseStream.Position != expected_GlyphStreamStartAt)
            {
                System.Diagnostics.Debug.WriteLine("ERR!!");
            }
#endif


            //***
            //some composite glyphs have instructions=> so we must check all composite glyphs
            //before read the glyph stream
            //** 
            using (MemoryStream compositeMS = new MemoryStream())
            {
                reader.BaseStream.Position = expected_CompositeStreamStartAt;
                compositeMS.Write(reader.ReadBytes((int)compositeStreamSize), 0, (int)compositeStreamSize);
                compositeMS.Position = 0;

                int j = compositeGlyphs.Count;
                ByteOrderSwappingBinaryReader compositeReader = new ByteOrderSwappingBinaryReader(compositeMS);
                for (ushort i = 0; i < j; ++i)
                {
                    ushort compositeGlyphIndex = compositeGlyphs[i];
                    allGlyphs[compositeGlyphIndex].compositeHasInstructions = CompositeHasInstructions(compositeReader, compositeGlyphIndex);
                }
                reader.BaseStream.Position = expected_GlyphStreamStartAt;
            }
            //-------- 
            int curFlagsIndex = 0;
            int pntContourIndex = 0;
            for (int i = 0; i < allGlyphs.Length; ++i)
            {
                readingGlyphs[i] = BuildSimpleGlyphStructure(reader,
                    ref allGlyphs[i],
                    emptyGlyph,
                    pntPerContours, ref pntContourIndex,
                    flagStream, ref curFlagsIndex);
            }

#if DEBUG
            if (pntContourIndex != pntPerContours.Length)
            {

            }
            if (curFlagsIndex != flagStream.Length)
            {

            }
#endif
            //--------------------------------------------------------------------------------------------
            //compositeStream
            //--------------------------------------------------------------------------------------------
#if DEBUG
            if (expected_CompositeStreamStartAt != reader.BaseStream.Position)
            {
                //***

                reader.BaseStream.Position = expected_CompositeStreamStartAt;
            }
#endif
            {
                //now we read the composite stream again
                //and create composite glyphs
                int j = compositeGlyphs.Count;
                for (ushort i = 0; i < j; ++i)
                {
                    int compositeGlyphIndex = compositeGlyphs[i];
                    readingGlyphs[compositeGlyphIndex] = ReadCompositeGlyph(readingGlyphs, reader, i, emptyGlyph);
                }
            }

            //--------------------------------------------------------------------------------------------
            //bbox stream
            //--------------------------------------------------------------------------------------------

            //Finally, for both simple and composite glyphs,
            //if the corresponding bit in the bounding box bit vector is set, 
            //then additionally read 4 Int16 values from the bbox stream, 
            //representing xMin, yMin, xMax, and yMax, respectively, 
            //and record these into the corresponding fields of the reconstructed glyph.
            //For simple glyphs, if the corresponding bit in the bounding box bit vector is not set,
            //then derive the bounding box by computing the minimum and maximum x and y coordinates in the outline, and storing that.

            //A composite glyph MUST have an explicitly supplied bounding box. 
            //The motivation is that computing bounding boxes is more complicated,
            //and would require resolving references to component glyphs taking into account composite glyph instructions and
            //the specified scales of individual components, which would conflict with a purely streaming implementation of font decoding.

            //A decoder MUST check for presence of the bounding box info as part of the composite glyph record 
            //and MUST NOT load a font file with the composite bounding box data missing. 
#if DEBUG
            if (expected_BboxStreamStartAt != reader.BaseStream.Position)
            {

            }
#endif
            int bitmapCount = (numGlyphs + 7) / 8;
            byte[] bboxBitmap = ExpandBitmap(reader.ReadBytes(bitmapCount));
            var glyphs = new Glyph[numGlyphs];
            for (ushort i = 0; i < numGlyphs; ++i)
            {
                TempGlyph tempGlyph = allGlyphs[i];
                var glyph = readingGlyphs[i] ?? throw new System.NotSupportedException
                    ($"Both {nameof(BuildSimpleGlyphStructure)} and {nameof(ReadCompositeGlyph)} failed to read the glyph at {i}");
                glyphs[i] = glyph;
                byte hasBbox = bboxBitmap[i];
                if (hasBbox == 1)
                {
                    //read bbox from the bboxstream
                    glyph.Bounds = new Bounds(
                        reader.ReadInt16(),
                        reader.ReadInt16(),
                        reader.ReadInt16(),
                        reader.ReadInt16());
                }
                else
                {
                    //no bbox
                    //
                    if (tempGlyph.numContour < 0)
                    {
                        //composite must have bbox
                        //if not=> err
                        throw new System.NotSupportedException();
                    }
                    else if (tempGlyph.numContour > 0)
                    {
                        //simple glyph
                        //use simple calculation
                        //...For simple glyphs, if the corresponding bit in the bounding box bit vector is not set,
                        //then derive the bounding box by computing the minimum and maximum x and y coordinates in the outline, and storing that.
                        glyph.Bounds = FindSimpleGlyphBounds(glyph);
                    }
                }
            }
            //--------------------------------------------------------------------------------------------
            //instruction stream
#if DEBUG
            if (reader.BaseStream.Position < expected_InstructionStreamStartAt)
            {

            }
            else if (expected_InstructionStreamStartAt == reader.BaseStream.Position)
            {

            }
            else
            {

            }
#endif

            reader.BaseStream.Position = expected_InstructionStreamStartAt;
            //--------------------------------------------------------------------------------------------

            for (ushort i = 0; i < numGlyphs; ++i)
            {
                TempGlyph tempGlyph = allGlyphs[i];
                if (tempGlyph.instructionLen > 0)
                {
                    glyphs[i].GlyphInstructions = reader.ReadBytes(tempGlyph.instructionLen);
                }
            }

#if DEBUG
            if (reader.BaseStream.Position != expected_EndAt)
            {

            }
#endif
            return new Glyf(header, glyphs, emptyGlyph);
        }

        static Bounds FindSimpleGlyphBounds(Glyph glyph)
        {
            if (!(glyph.TtfWoffInfo is var (_, glyphPoints)))
            {
                throw new System.NotImplementedException("Built glyph is not WOFF glyph");
            }

            int j = glyphPoints.Length;
            float xmin = float.MaxValue;
            float ymin = float.MaxValue;
            float xmax = float.MinValue;
            float ymax = float.MinValue;

            for (int i = 0; i < j; ++i)
            {
                GlyphPointF p = glyphPoints[i];
                if (p.X < xmin)
                {
                    xmin = p.X;
                }
                if (p.X > xmax)
                {
                    xmax = p.X;
                }
                if (p.Y < ymin)
                {
                    ymin = p.Y;
                }
                if (p.Y > ymax)
                {
                    ymax = p.Y;
                }
            }

            return new Bounds(
               (short)System.Math.Round(xmin),
               (short)System.Math.Round(ymin),
               (short)System.Math.Round(xmax),
               (short)System.Math.Round(ymax));
        }

        static byte[] ExpandBitmap(byte[] orgBBoxBitmap)
        {
            byte[] expandArr = new byte[orgBBoxBitmap.Length * 8];

            int index = 0;
            for (int i = 0; i < orgBBoxBitmap.Length; ++i)
            {
                byte b = orgBBoxBitmap[i];
                expandArr[index++] = (byte)((b >> 7) & 0x1);
                expandArr[index++] = (byte)((b >> 6) & 0x1);
                expandArr[index++] = (byte)((b >> 5) & 0x1);
                expandArr[index++] = (byte)((b >> 4) & 0x1);
                expandArr[index++] = (byte)((b >> 3) & 0x1);
                expandArr[index++] = (byte)((b >> 2) & 0x1);
                expandArr[index++] = (byte)((b >> 1) & 0x1);
                expandArr[index++] = (byte)((b >> 0) & 0x1);
            }
            return expandArr;
        }

        static Glyph? BuildSimpleGlyphStructure(BinaryReader glyphStreamReader,
            ref TempGlyph tmpGlyph,
            Glyph emptyGlyph,
            ushort[] pntPerContours, ref int pntContourIndex,
            byte[] flagStream, ref int flagStreamIndex)
        {
            if (tmpGlyph.numContour == 0)
            {
                return emptyGlyph;
            }
            if (tmpGlyph.numContour < 0)
            {
                //composite glyph,
                //check if this has instruction or not
                if (tmpGlyph.compositeHasInstructions)
                {
                    tmpGlyph.instructionLen = Woff2Utils.Read255UInt16(glyphStreamReader);
                }
                return null;//skip composite glyph (resolve later)     
            }

            //-----
            int curX = 0;
            int curY = 0;

            int numContour = tmpGlyph.numContour;

            var _endContours = new ushort[numContour];
            ushort pointCount = 0;

            //create contours
            for (ushort i = 0; i < numContour; ++i)
            {
                ushort numPoint = pntPerContours[pntContourIndex++];//increament pntContourIndex AFTER
                pointCount += numPoint;
                _endContours[i] = (ushort)(pointCount - 1);
            }

            //collect point for our contours
            var _glyphPoints = new GlyphPointF[pointCount];
            int n = 0;
            for (int i = 0; i < numContour; ++i)
            {
                //read point detail
                //step 3) 

                //foreach contour
                //read 1 byte flags for each contour

                //1) The most significant bit of a flag indicates whether the point is on- or off-curve point,
                //2) the remaining seven bits of the flag determine the format of X and Y coordinate values and 
                //specify 128 possible combinations of indices that have been assigned taking into consideration 
                //typical statistical distribution of data found in TrueType fonts. 

                //When X and Y coordinate values are recorded using nibbles(either 4 bits per coordinate or 12 bits per coordinate)
                //the bits are packed in the byte stream with most significant bit of X coordinate first, 
                //followed by the value for Y coordinate (most significant bit first). 
                //As a result, the size of the glyph dataset is significantly reduced, 
                //and the grouping of the similar values(flags, coordinates) in separate and contiguous data streams allows 
                //more efficient application of the entropy coding applied as the second stage of encoding process. 

                int endContour = _endContours[i];
                for (; n <= endContour; ++n)
                {

                    byte f = flagStream[flagStreamIndex++]; //increment the flagStreamIndex AFTER read

                    //int f1 = (f >> 7); // most significant 1 bit -> on/off curve

                    int xyFormat = f & 0x7F; // remainging 7 bits x,y format  

                    TripleEncodingRecord enc = s_encTable[xyFormat]; //0-128 

                    byte[] packedXY = glyphStreamReader.ReadBytes(enc.ByteCount - 1); //byte count include 1 byte flags, so actual read=> byteCount-1
                                                                                      //read x and y 

                    int x = 0;
                    int y = 0;

                    switch (enc.XBits)
                    {
                        default:
                            throw new System.NotSupportedException();//???
                        case 0: //0,8, 
                            x = 0;
                            y = enc.Ty(packedXY[0]);
                            break;
                        case 4: //4,4
                            x = enc.Tx(packedXY[0] >> 4);
                            y = enc.Ty(packedXY[0] & 0xF);
                            break;
                        case 8: //8,0 or 8,8
                            x = enc.Tx(packedXY[0]);
                            y = (enc.YBits == 8) ?
                                    enc.Ty(packedXY[1]) :
                                    0;
                            break;
                        case 12:
                            x = enc.Tx((packedXY[0] << 4) | (packedXY[1] >> 4));
                            y = enc.Ty(((packedXY[1] & 0xF) << 8) | (packedXY[2]));
                            break;
                        case 16: //16,16
                            x = enc.Tx((packedXY[0] << 8) | packedXY[1]);
                            y = enc.Ty((packedXY[2] << 8) | packedXY[3]);
                            break;
                    }

                    //incremental point format***
                    _glyphPoints[n] = new GlyphPointF(curX += x, curY += y, (f >> 7) == 0); // most significant 1 bit -> on/off curve 
                }
            }

            //----
            //step 4) Read one 255UInt16 value from the glyph stream, which is instructionLength, the number of instruction bytes.
            tmpGlyph.instructionLen = Woff2Utils.Read255UInt16(glyphStreamReader);
            //step 5) resolve it later

            return new Glyph(_glyphPoints,
               _endContours,
               new Bounds(), //calculate later
               null,  //load instruction later
               tmpGlyph.glyphIndex);
        }

        static bool CompositeHasInstructions(BinaryReader reader, ushort compositeGlyphIndex)
        {

            //To find if a composite has instruction or not.

            //This method is similar to  ReadCompositeGlyph() (below)
            //but this dose not create actual composite glyph.

            Glyf.CompositeGlyphFlags flags;
            do
            {
                flags = (Glyf.CompositeGlyphFlags)reader.ReadUInt16();
                short arg1 = 0;
                short arg2 = 0;
                ushort arg1and2 = 0;

                if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.ARG_1_AND_2_ARE_WORDS))
                {
                    arg1 = reader.ReadInt16();
                    arg2 = reader.ReadInt16();
                }
                else
                {
                    arg1and2 = reader.ReadUInt16();
                }
                //-----------------------------------------
                float xscale = 1;
                float scale01 = 0;
                float scale10 = 0;
                float yscale = 1;

                bool useMatrix = false;
                //-----------------------------------------
                bool hasScale = false;
                if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_A_SCALE))
                {
                    //If the bit WE_HAVE_A_SCALE is set,
                    //the scale value is read in 2.14 format-the value can be between -2 to almost +2.
                    //The glyph will be scaled by this value before grid-fitting. 
                    xscale = yscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    hasScale = true;
                }
                else if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_AN_X_AND_Y_SCALE))
                {
                    xscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    yscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    hasScale = true;
                }
                else if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_A_TWO_BY_TWO))
                {

                    //The bit WE_HAVE_A_TWO_BY_TWO allows for linear transformation of the X and Y coordinates by specifying a 2 × 2 matrix.
                    //This could be used for scaling and 90-degree*** rotations of the glyph components, for example.

                    //2x2 matrix

                    //The purpose of USE_MY_METRICS is to force the lsb and rsb to take on a desired value.
                    //For example, an i-circumflex (U+00EF) is often composed of the circumflex and a dotless-i. 
                    //In order to force the composite to have the same metrics as the dotless-i,
                    //set USE_MY_METRICS for the dotless-i component of the composite. 
                    //Without this bit, the rsb and lsb would be calculated from the hmtx entry for the composite 
                    //(or would need to be explicitly set with TrueType instructions).

                    //Note that the behavior of the USE_MY_METRICS operation is undefined for rotated composite components. 
                    useMatrix = true;
                    hasScale = true;
                    xscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    scale01 = reader.ReadF2Dot14(); /* Format 2.14 */
                    scale10 = reader.ReadF2Dot14();/* Format 2.14 */
                    yscale = reader.ReadF2Dot14(); /* Format 2.14 */

                }

            } while (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.MORE_COMPONENTS));

            //
            return Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_INSTRUCTIONS);
        }

        static Glyph ReadCompositeGlyph(Glyph?[] createdGlyphs, BinaryReader reader, ushort compositeGlyphIndex, Glyph emptyGlyph)
        {

            //Decoding of Composite Glyphs
            //For a composite glyph(nContour == -1), the following steps take the place of (Building Simple Glyph, steps 1 - 5 above):

            //1a.Read a UInt16 from compositeStream.
            //  This is interpreted as a component flag word as in the TrueType spec.
            //  Based on the flag values, there are between 4 and 14 additional argument bytes,
            //  interpreted as glyph index, arg1, arg2, and optional scale or affine matrix.

            //2a.Read the number of argument bytes as determined in step 2a from the composite stream, 
            //and store these in the reconstructed glyph.
            //If the flag word read in step 2a has the FLAG_MORE_COMPONENTS bit(bit 5) set, go back to step 2a.

            //3a.If any of the flag words had the FLAG_WE_HAVE_INSTRUCTIONS bit(bit 8) set,
            //then read the instructions from the glyph and store them in the reconstructed glyph, 
            //using the same process as described in steps 4 and 5 above (see Building Simple Glyph).



            Glyph? finalGlyph = null;
            Glyf.CompositeGlyphFlags flags;
            do
            {
                flags = (Glyf.CompositeGlyphFlags)reader.ReadUInt16();
                ushort glyphIndex = reader.ReadUInt16();
                var existingGlyph = createdGlyphs[glyphIndex];
                if (existingGlyph == null)
                {
                    // This glyph is not read yet, resolve it first!
                    long storedOffset = reader.BaseStream.Position;
                    Glyph missingGlyph = ReadCompositeGlyph(createdGlyphs, reader, glyphIndex, emptyGlyph);
                    createdGlyphs[glyphIndex] = existingGlyph = missingGlyph;
                    reader.BaseStream.Position = storedOffset;
                }

                Glyph newGlyph = Glyph.Clone(existingGlyph, compositeGlyphIndex);

                short arg1 = 0;
                short arg2 = 0;
                ushort arg1and2 = 0;

                if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.ARG_1_AND_2_ARE_WORDS))
                {
                    arg1 = reader.ReadInt16();
                    arg2 = reader.ReadInt16();
                }
                else
                {
                    arg1and2 = reader.ReadUInt16();
                }
                //-----------------------------------------
                float xscale = 1;
                float scale01 = 0;
                float scale10 = 0;
                float yscale = 1;

                bool useMatrix = false;
                //-----------------------------------------
                bool hasScale = false;
                if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_A_SCALE))
                {
                    //If the bit WE_HAVE_A_SCALE is set,
                    //the scale value is read in 2.14 format-the value can be between -2 to almost +2.
                    //The glyph will be scaled by this value before grid-fitting. 
                    xscale = yscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    hasScale = true;
                }
                else if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_AN_X_AND_Y_SCALE))
                {
                    xscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    yscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    hasScale = true;
                }
                else if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.WE_HAVE_A_TWO_BY_TWO))
                {

                    //The bit WE_HAVE_A_TWO_BY_TWO allows for linear transformation of the X and Y coordinates by specifying a 2 × 2 matrix.
                    //This could be used for scaling and 90-degree*** rotations of the glyph components, for example.

                    //2x2 matrix

                    //The purpose of USE_MY_METRICS is to force the lsb and rsb to take on a desired value.
                    //For example, an i-circumflex (U+00EF) is often composed of the circumflex and a dotless-i. 
                    //In order to force the composite to have the same metrics as the dotless-i,
                    //set USE_MY_METRICS for the dotless-i component of the composite. 
                    //Without this bit, the rsb and lsb would be calculated from the hmtx entry for the composite 
                    //(or would need to be explicitly set with TrueType instructions).

                    //Note that the behavior of the USE_MY_METRICS operation is undefined for rotated composite components. 
                    useMatrix = true;
                    hasScale = true;
                    xscale = reader.ReadF2Dot14(); /* Format 2.14 */
                    scale01 = reader.ReadF2Dot14(); /* Format 2.14 */
                    scale10 = reader.ReadF2Dot14(); /* Format 2.14 */
                    yscale = reader.ReadF2Dot14(); /* Format 2.14 */

                    if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.UNSCALED_COMPONENT_OFFSET))
                    {


                    }
                    else
                    {


                    }
                    if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.USE_MY_METRICS))
                    {

                    }
                }

                //--------------------------------------------------------------------
                if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.ARGS_ARE_XY_VALUES))
                {
                    //Argument1 and argument2 can be either x and y offsets to be added to the glyph or two point numbers.  
                    //x and y offsets to be added to the glyph
                    //When arguments 1 and 2 are an x and a y offset instead of points and the bit ROUND_XY_TO_GRID is set to 1,
                    //the values are rounded to those of the closest grid lines before they are added to the glyph.
                    //X and Y offsets are described in FUnits. 

                    if (useMatrix)
                    {
                        //use this matrix  
                        Glyph.TransformNormalWith2x2Matrix(newGlyph, xscale, scale01, scale10, yscale);
                        Glyph.OffsetXY(newGlyph, arg1, arg2);
                    }
                    else
                    {
                        if (hasScale)
                        {
                            if (xscale == 1.0 && yscale == 1.0)
                            {

                            }
                            else
                            {
                                Glyph.TransformNormalWith2x2Matrix(newGlyph, xscale, 0, 0, yscale);
                            }
                            Glyph.OffsetXY(newGlyph, arg1, arg2);
                        }
                        else
                        {
                            if (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.ROUND_XY_TO_GRID))
                            {
                                //TODO: implement round xy to grid***
                                //----------------------------
                            }
                            //just offset***
                            Glyph.OffsetXY(newGlyph, arg1, arg2);
                        }
                    }


                }
                else
                {
                    //two point numbers. 
                    //the first point number indicates the point that is to be matched to the new glyph. 
                    //The second number indicates the new glyph's “matched” point. 
                    //Once a glyph is added,its point numbers begin directly after the last glyphs (endpoint of first glyph + 1)

                }

                //
                if (finalGlyph == null)
                {
                    finalGlyph = newGlyph;
                }
                else
                {
                    //merge 
                    Glyph.AppendGlyph(finalGlyph, newGlyph);
                }

            } while (Glyf.HasFlag(flags, Glyf.CompositeGlyphFlags.MORE_COMPONENTS));

            return finalGlyph ?? emptyGlyph;
        }

        struct TripleEncodingRecord
        {
            public readonly byte ByteCount;
            public readonly byte XBits;
            public readonly byte YBits;
            public readonly ushort DeltaX;
            public readonly ushort DeltaY;
            public readonly sbyte Xsign;
            public readonly sbyte Ysign;

            public TripleEncodingRecord(
                byte byteCount,
                byte xbits, byte ybits,
                ushort deltaX, ushort deltaY,
                sbyte xsign, sbyte ysign)
            {
                ByteCount = byteCount;
                XBits = xbits;
                YBits = ybits;
                DeltaX = deltaX;
                DeltaY = deltaY;
                Xsign = xsign;
                Ysign = ysign;
#if DEBUG
                debugIndex = -1;
#endif
            }
#if DEBUG
            public int debugIndex;
            public override string ToString()
            {
                return debugIndex + " " + ByteCount + " " + XBits + " " + YBits + " " + DeltaX + " " + DeltaY + " " + Xsign + " " + Ysign;
            }
#endif
            /// <summary>
            /// translate X
            /// </summary>
            /// <param name="orgX"></param>
            /// <returns></returns>
            public int Tx(int orgX) => (orgX + DeltaX) * Xsign;

            /// <summary>
            /// translate Y
            /// </summary>
            /// <param name="orgY"></param>
            /// <returns></returns>
            public int Ty(int orgY) => (orgY + DeltaY) * Ysign;

        }

        class TripleEncodingTable
        {

            static TripleEncodingTable? s_encTable;

            readonly List<TripleEncodingRecord> _records = new List<TripleEncodingRecord>();
            public static TripleEncodingTable GetEncTable() => s_encTable ??= new TripleEncodingTable();
            private TripleEncodingTable()
            {

                BuildTable();

#if DEBUG
                if (_records.Count != 128)
                {
                    throw new System.Exception();
                }
                dbugValidateTable();
#endif
            }
#if DEBUG
            void dbugValidateTable()
            {
#if DEBUG
                for (int xyFormat = 0; xyFormat < 128; ++xyFormat)
                {
                    TripleEncodingRecord tripleRec = _records[xyFormat];
                    if (xyFormat < 84)
                    {
                        //0-83 inclusive
                        if ((tripleRec.ByteCount - 1) != 1)
                        {
                            throw new System.NotSupportedException();
                        }
                    }
                    else if (xyFormat < 120)
                    {
                        //84-119 inclusive
                        if ((tripleRec.ByteCount - 1) != 2)
                        {
                            throw new System.NotSupportedException();
                        }
                    }
                    else if (xyFormat < 124)
                    {
                        //120-123 inclusive
                        if ((tripleRec.ByteCount - 1) != 3)
                        {
                            throw new System.NotSupportedException();
                        }
                    }
                    else if (xyFormat < 128)
                    {
                        //124-127 inclusive
                        if ((tripleRec.ByteCount - 1) != 4)
                        {
                            throw new System.NotSupportedException();
                        }
                    }
                }

#endif
            }
#endif
            public TripleEncodingRecord this[int index] => _records[index];

            void BuildTable()
            {
                BuildRecords(2, 0, 8, null, new ushort[] { 0, 256, 512, 768, 1024 }); //2*5

                BuildRecords(2, 8, 0, new ushort[] { 0, 256, 512, 768, 1024 }, null); //2*5

                BuildRecords(2, 4, 4, new ushort[] { 1 }, new ushort[] { 1, 17, 33, 49 });// 4*4 => 16 records

                BuildRecords(2, 4, 4, new ushort[] { 17 }, new ushort[] { 1, 17, 33, 49 });// 4*4 => 16 records

                BuildRecords(2, 4, 4, new ushort[] { 33 }, new ushort[] { 1, 17, 33, 49 });// 4*4 => 16 records

                BuildRecords(2, 4, 4, new ushort[] { 49 }, new ushort[] { 1, 17, 33, 49 });// 4*4 => 16 records

                BuildRecords(3, 8, 8, new ushort[] { 1 }, new ushort[] { 1, 257, 513 });// 4*3 => 12 records

                BuildRecords(3, 8, 8, new ushort[] { 257 }, new ushort[] { 1, 257, 513 });// 4*3 => 12 records

                BuildRecords(3, 8, 8, new ushort[] { 513 }, new ushort[] { 1, 257, 513 });// 4*3 => 12 records

                BuildRecords(4, 12, 12, new ushort[] { 0 }, new ushort[] { 0 }); // 4*1 => 4 records

                BuildRecords(5, 16, 16, new ushort[] { 0 }, new ushort[] { 0 });// 4*1 => 4 records

            }
            void AddRecord(byte byteCount, byte xbits, byte ybits, ushort deltaX, ushort deltaY, sbyte xsign, sbyte ysign)
            {
                var rec = new TripleEncodingRecord(byteCount, xbits, ybits, deltaX, deltaY, xsign, ysign);
#if DEBUG
                rec.debugIndex = _records.Count;
#endif
                _records.Add(rec);
            }
            void BuildRecords(byte byteCount, byte xbits, byte ybits, ushort[]? deltaXs, ushort[]? deltaYs)
            {
                if (deltaXs == null)
                {
                    if (deltaYs == null)
                    {
                        throw new System.ArgumentException($"{nameof(deltaXs)} and {nameof(deltaYs)} cannot both be null");
                    }
                    //(set 1.1)
                    for (int y = 0; y < deltaYs.Length; ++y)
                    {
                        AddRecord(byteCount, xbits, ybits, 0, deltaYs[y], 0, -1);
                        AddRecord(byteCount, xbits, ybits, 0, deltaYs[y], 0, 1);
                    }
                }
                else if (deltaYs == null)
                {
                    //(set 1.2)
                    for (int x = 0; x < deltaXs.Length; ++x)
                    {
                        AddRecord(byteCount, xbits, ybits, deltaXs[x], 0, -1, 0);
                        AddRecord(byteCount, xbits, ybits, deltaXs[x], 0, 1, 0);
                    }
                }
                else
                {
                    //set 2.1, - set5
                    for (int x = 0; x < deltaXs.Length; ++x)
                    {
                        ushort deltaX = deltaXs[x];

                        for (int y = 0; y < deltaYs.Length; ++y)
                        {
                            ushort deltaY = deltaYs[y];

                            AddRecord(byteCount, xbits, ybits, deltaX, deltaY, -1, -1);
                            AddRecord(byteCount, xbits, ybits, deltaX, deltaY, 1, -1);
                            AddRecord(byteCount, xbits, ybits, deltaX, deltaY, -1, 1);
                            AddRecord(byteCount, xbits, ybits, deltaX, deltaY, 1, 1);
                        }
                    }
                }
            }
        }
    }

    class TransformedLoca : UnreadTableEntry
    {
        public TransformedLoca(TableHeader header, Woff2TableDirectory tableDir) : base(header)
        {
            HasCustomContentReader = true;
            TableDir = tableDir;
        }
        public Woff2TableDirectory TableDir { get; }
        public override T CreateTableEntry<T>(BinaryReader reader, OpenFontReader.TableReader<T> expectedResult)
        {
            //nothing todo here, read nothing :)
            return
                new GlyphLocations(-1, true, Header, new BinaryReader(Stream.Null)) is T t
                ? t : throw new System.NotSupportedException();
        }

    }

    class Woff2Reader
    {

        public BrotliDecompressStreamFunc? DecompressHandler;

        public Woff2Reader()
        {
#if DEBUG
            dbugVerifyKnownTables();
#endif
        }
#if DEBUG

        static bool s_dbugPassVeriKnownTables;
        static void dbugVerifyKnownTables()
        {
            if (s_dbugPassVeriKnownTables)
            {
                return;
            }
            //--------------
            Dictionary<string, bool> uniqueNames = new Dictionary<string, bool>();
            foreach (string name in s_knownTableTags)
            {
                if (!uniqueNames.ContainsKey(name))
                {
                    uniqueNames.Add(name, true);
                }
                else
                {
                    throw new System.Exception();
                }
            }
        }
#endif

        public PreviewFontInfo? ReadPreview(BinaryReader reader)
        {
            var header = ReadHeader(reader);
            if (header == null)
            {
                return null;  //=> return here and notify user too.
            }
            Woff2TableDirectory[] woff2TablDirs = ReadTableDirectories(header, reader);
            if (DecompressHandler == null)
            {
                //if no Brotli decoder=> return here and notify user too.
                if (Woff2DefaultBrotliDecompressFunc.DecompressHandler != null)
                {
                    DecompressHandler = Woff2DefaultBrotliDecompressFunc.DecompressHandler;
                }
                else
                {
                    //return here and notify user too. 
                    return null;
                }
            }

            //try read each compressed tables
            byte[] compressedBuffer = reader.ReadBytes((int)header.totalCompressSize);
            if (compressedBuffer.Length != header.totalCompressSize)
            {
                //error!
                return null; //can't read this, notify user too.
            }
            using (MemoryStream decompressedStream = new MemoryStream())
            {
                if (!DecompressHandler(compressedBuffer, decompressedStream))
                {
                    return null;
                }
                //from decoded stream we read each table
                decompressedStream.Position = 0;//reset pos

                using (ByteOrderSwappingBinaryReader reader2 = new ByteOrderSwappingBinaryReader(decompressedStream))
                {
                    TableEntryCollection tableEntryCollection = CreateTableEntryCollection(woff2TablDirs);
                    OpenFontReader openFontReader = new OpenFontReader();
                    return openFontReader.ReadPreviewFontInfo(tableEntryCollection, reader2);
                }
            }
        }
        internal Typeface? Read(BinaryReader reader)
        {
            var header = ReadHeader(reader);
            if (header == null) return null;  //=> return here and notify user too.
            Woff2TableDirectory[] woff2TablDirs = ReadTableDirectories(header, reader);
            if (DecompressHandler == null)
            {
                //if no Brotli decoder=> return here and notify user too.
                if (Woff2DefaultBrotliDecompressFunc.DecompressHandler != null)
                {
                    DecompressHandler = Woff2DefaultBrotliDecompressFunc.DecompressHandler;
                }
                else
                {
                    //return here and notify user too. 
                    return null;
                }
            }

            //try read each compressed tables
            byte[] compressedBuffer = reader.ReadBytes((int)header.totalCompressSize);
            if (compressedBuffer.Length != header.totalCompressSize)
            {
                //error!
                return null; //can't read this, notify user too.
            }

            using (MemoryStream decompressedStream = new MemoryStream())
            {
                if (!DecompressHandler(compressedBuffer, decompressedStream))
                {
                    //...Most notably, 
                    //the data for the font tables is compressed in a SINGLE data stream comprising all the font tables.

                    //if not pass set to null
                    //decompressedBuffer = null;
                    return null;
                }
                //from decoded stream we read each table
                decompressedStream.Position = 0;//reset pos

                using (ByteOrderSwappingBinaryReader reader2 = new ByteOrderSwappingBinaryReader(decompressedStream))
                {
                    TableEntryCollection tableEntryCollection = CreateTableEntryCollection(woff2TablDirs);
                    OpenFontReader openFontReader = new OpenFontReader();
                    return openFontReader.ReadTableEntryCollection(tableEntryCollection, reader2);
                }
            }
        }
        public Typeface? Read(Stream inputstream)
        {
            using (ByteOrderSwappingBinaryReader reader = new ByteOrderSwappingBinaryReader(inputstream))
            {
                return Read(reader);
            }
        }



        Woff2Header? ReadHeader(BinaryReader reader)
        {
            //WOFF2 Header
            //UInt32  signature             0x774F4632 'wOF2'
            //UInt32  flavor                The "sfnt version" of the input font.
            //UInt32  length                Total size of the WOFF file.
            //UInt16  numTables             Number of entries in directory of font tables.
            //UInt16  reserved              Reserved; set to 0.
            //UInt32  totalSfntSize         Total size needed for the uncompressed font data, including the sfnt header,
            //                              directory, and font tables(including padding).
            //UInt32  totalCompressedSize   Total length of the compressed data block.
            //UInt16  majorVersion          Major version of the WOFF file.
            //UInt16  minorVersion          Minor version of the WOFF file.
            //UInt32  metaOffset            Offset to metadata block, from beginning of WOFF file.
            //UInt32  metaLength            Length of compressed metadata block.
            //UInt32  metaOrigLength        Uncompressed size of metadata block.
            //UInt32  privOffset            Offset to private data block, from beginning of WOFF file.
            //UInt32  privLength            Length of private data block.

            Woff2Header header = new Woff2Header();
            byte b0 = reader.ReadByte();
            byte b1 = reader.ReadByte();
            byte b2 = reader.ReadByte();
            byte b3 = reader.ReadByte();
            if (!(b0 == 0x77 && b1 == 0x4f && b2 == 0x46 && b3 == 0x32))
            {
                return null;
            }
            header.flavor = reader.ReadUInt32();

            header.length = reader.ReadUInt32();
            header.numTables = reader.ReadUInt16();
            header.totalSfntSize = reader.ReadUInt32();
            header.totalCompressSize = reader.ReadUInt32();//***

            header.majorVersion = reader.ReadUInt16();
            header.minorVersion = reader.ReadUInt16();

            header.metaOffset = reader.ReadUInt32();
            header.metaLength = reader.ReadUInt32();
            header.metaOriginalLength = reader.ReadUInt32();

            header.privOffset = reader.ReadUInt32();
            header.privLength = reader.ReadUInt32();

            return header;
        }

        Woff2TableDirectory[] ReadTableDirectories(Woff2Header header, BinaryReader reader)
        {

            uint tableCount = header.numTables; //?
            var tableDirs = new Woff2TableDirectory[tableCount];

            long expectedTableStartAt = 0;

            for (int i = 0; i < tableCount; ++i)
            {
                //TableDirectoryEntry
                //UInt8         flags           table type and flags
                //UInt32        tag	            4-byte tag(optional)
                //UIntBase128   origLength      length of original table
                //UIntBase128   transformLength transformed length(if applicable)

                byte flags = reader.ReadByte();
                //The interpretation of the flags field is as follows.

                //Bits[0..5] contain an index to the "known tag" table, 
                //which represents tags likely to appear in fonts.If the tag is not present in this table,
                //then the value of this bit field is 63. 

                //interprete flags 
                int knowTable = flags & 0x1F; //5 bits => known table or not  
                string tableName;
                if (knowTable < 63)
                {
                    //this is known table
                    tableName = s_knownTableTags[knowTable];
                }
                else
                {
                    tableName = Utils.TagToString(reader.ReadUInt32()); //other tag 
                }
                Woff2TableDirectory table = new Woff2TableDirectory(tableName);

                //Bits 6 and 7 indicate the preprocessing transformation version number(0 - 3) that was applied to each table.

                //For all tables in a font, except for 'glyf' and 'loca' tables,
                //transformation version 0 indicates the null transform where the original table data is passed directly 
                //to the Brotli compressor for inclusion in the compressed data stream.

                //For 'glyf' and 'loca' tables,
                //transformation version 3 indicates the null transform where the original table data was passed directly 
                //to the Brotli compressor without applying any pre - processing defined in subclause 5.1 and subclause 5.3.

                //The transformed table formats and their associated transformation version numbers are 
                //described in details in clause 5 of this specification.


                table.PreprocessingTransformation = (byte)((flags >> 5) & 0x3); //2 bits, preprocessing transformation


                table.ExpectedStartAt = expectedTableStartAt;
                //
                if (!ReadUIntBase128(reader, out table.origLength))
                {
                    //can't read 128=> error
                }

                switch (table.PreprocessingTransformation)
                {
                    default:
                        Console.WriteLine("Default");
                        break;
                    case 0:
                        {
                            if (table.Name == Glyf.Name)
                            {
                                if (!ReadUIntBase128(reader, out table.transformLength))
                                {
                                    //can't read 128=> error
                                }
                                expectedTableStartAt += table.transformLength;//***
                            }
                            else if (table.Name == GlyphLocations.Name)
                            {
                                //BUT by spec, transform 'loca' MUST has transformLength=0
                                if (!ReadUIntBase128(reader, out table.transformLength))
                                {
                                    //can't read 128=> error
                                }
                                expectedTableStartAt += table.transformLength;//***
                            }
                            else
                            {
                                expectedTableStartAt += table.origLength;
                            }
                        }
                        break;
                    case 1:
                        {
                            expectedTableStartAt += table.origLength;
                        }
                        break;
                    case 2:
                        {
                            expectedTableStartAt += table.origLength;
                        }
                        break;
                    case 3:
                        {
                            expectedTableStartAt += table.origLength;
                        }
                        break;
                }
                tableDirs[i] = table;
            }

            return tableDirs;
        }

        static TableEntryCollection CreateTableEntryCollection(Woff2TableDirectory[] woffTableDirs)
        {
            TableEntryCollection tableEntryCollection = new TableEntryCollection();
            for (int i = 0; i < woffTableDirs.Length; ++i)
            {
                Woff2TableDirectory woffTableDir = woffTableDirs[i];
                UnreadTableEntry unreadTableEntry;

                if (woffTableDir.Name == Glyf.Name && woffTableDir.PreprocessingTransformation == 0)
                {
                    //this is transformed glyf table,
                    //we need another techqniue 
                    TableHeader tableHeader = new TableHeader(woffTableDir.Name, 0,
                                       (uint)woffTableDir.ExpectedStartAt,
                                       woffTableDir.transformLength);
                    unreadTableEntry = new TransformedGlyf(tableHeader, woffTableDir);

                }
                else if (woffTableDir.Name == GlyphLocations.Name && woffTableDir.PreprocessingTransformation == 0)
                {
                    //this is transformed glyf table,
                    //we need another techqniue 
                    TableHeader tableHeader = new TableHeader(woffTableDir.Name, 0,
                                       (uint)woffTableDir.ExpectedStartAt,
                                       woffTableDir.transformLength);
                    unreadTableEntry = new TransformedLoca(tableHeader, woffTableDir);
                }
                else
                {
                    TableHeader tableHeader = new TableHeader(woffTableDir.Name, 0,
                                          (uint)woffTableDir.ExpectedStartAt,
                                          woffTableDir.origLength);
                    unreadTableEntry = new UnreadTableEntry(tableHeader);
                }
                tableEntryCollection.AddEntry(unreadTableEntry.Name, unreadTableEntry);
            }

            return tableEntryCollection;
        }


        static readonly string[] s_knownTableTags = new string[]
        {
             //Known Table Tags
            //Flag  Tag         Flag  Tag       Flag  Tag        Flag    Tag
            //0	 => cmap,	    16 =>EBLC,	    32 =>CBDT,	     48 =>gvar,
            //1  => head,	    17 =>gasp,	    33 =>CBLC,	     49 =>hsty,
            //2	 => hhea,	    18 =>hdmx,	    34 =>COLR,	     50 =>just,
            //3	 => hmtx,	    19 =>kern,	    35 =>CPAL,	     51 =>lcar,
            //4	 => maxp,	    20 =>LTSH,	    36 =>SVG ,	     52 =>mort,
            //5	 => name,	    21 =>PCLT,	    37 =>sbix,	     53 =>morx,
            //6	 => OS/2,	    22 =>VDMX,	    38 =>acnt,	     54 =>opbd,
            //7	 => post,	    23 =>vhea,	    39 =>avar,	     55 =>prop,
            //8	 => cvt ,	    24 =>vmtx,	    40 =>bdat,	     56 =>trak,
            //9	 => fpgm,	    25 =>BASE,	    41 =>bloc,	     57 =>Zapf,
            //10 =>	glyf,	    26 =>GDEF,	    42 =>bsln,	     58 =>Silf,
            //11 =>	loca,	    27 =>GPOS,	    43 =>cvar,	     59 =>Glat,
            //12 =>	prep,	    28 =>GSUB,	    44 =>fdsc,	     60 =>Gloc,
            //13 =>	CFF ,	    29 =>EBSC,	    45 =>feat,	     61 =>Feat,
            //14 =>	VORG,	    30 =>JSTF,	    46 =>fmtx,	     62 =>Sill,
            //15 =>	EBDT,	    31 =>MATH,	    47 =>fvar,	     63 =>arbitrary tag follows,...
            //-------------------------------------------------------------------

            //-- TODO:implement missing table too!
            Cmap.Name, //0
            Head.Name, //1
            HorizontalHeader.Name,//2
            HorizontalMetrics.Name,//3
            MaxProfile.Name,//4
            NameEntry.Name,//5
            OS2Table.Name, //6
            PostTable.Name,//7
            CvtTable.Name,//8
            FpgmTable.Name,//9
            Glyf.Name,//10
            GlyphLocations.Name,//11
            PrepTable.Name,//12
            CFFTable.Name,//13
            "VORG",//14 
            EBDT.Name,//15, 

            
            //---------------
            EBLC.Name,//16
            Gasp.Name,//17
            HorizontalDeviceMetrics.Name,//18
            Kern.Name,//19
            "LTSH",//20 
            "PCLT",//21
            VerticalDeviceMetrics.Name,//22
            VerticalHeader.Name,//23
            VerticalMetrics.Name,//24
            BASE.Name,//25
            GDEF.Name,//26
            GPOS.Name,//27
            GSUB.Name,//28            
            EBSC.Name, //29
            "JSTF", //30
            MathTable.Name,//31
             //---------------


            //Known Table Tags (copy,same as above)
            //Flag  Tag         Flag  Tag       Flag  Tag        Flag    Tag
            //0	 => cmap,	    16 =>EBLC,	    32 =>CBDT,	     48 =>gvar,
            //1  => head,	    17 =>gasp,	    33 =>CBLC,	     49 =>hsty,
            //2	 => hhea,	    18 =>hdmx,	    34 =>COLR,	     50 =>just,
            //3	 => hmtx,	    19 =>kern,	    35 =>CPAL,	     51 =>lcar,
            //4	 => maxp,	    20 =>LTSH,	    36 =>SVG ,	     52 =>mort,
            //5	 => name,	    21 =>PCLT,	    37 =>sbix,	     53 =>morx,
            //6	 => OS/2,	    22 =>VDMX,	    38 =>acnt,	     54 =>opbd,
            //7	 => post,	    23 =>vhea,	    39 =>avar,	     55 =>prop,
            //8	 => cvt ,	    24 =>vmtx,	    40 =>bdat,	     56 =>trak,
            //9	 => fpgm,	    25 =>BASE,	    41 =>bloc,	     57 =>Zapf,
            //10 =>	glyf,	    26 =>GDEF,	    42 =>bsln,	     58 =>Silf,
            //11 =>	loca,	    27 =>GPOS,	    43 =>cvar,	     59 =>Glat,
            //12 =>	prep,	    28 =>GSUB,	    44 =>fdsc,	     60 =>Gloc,
            //13 =>	CFF ,	    29 =>EBSC,	    45 =>feat,	     61 =>Feat,
            //14 =>	VORG,	    30 =>JSTF,	    46 =>fmtx,	     62 =>Sill,
            //15 =>	EBDT,	    31 =>MATH,	    47 =>fvar,	     63 =>arbitrary tag follows,...
            //-------------------------------------------------------------------

            CBDT.Name, //32
            CBLC.Name,//33
            COLR.Name,//34
            CPAL.Name,//35,
            SvgTable.Name,//36
            "sbix",//37
            "acnt",//38
            "avar",//39
            "bdat",//40
            "bloc",//41
            "bsln",//42
            "cvar",//43
            "fdsc",//44
            "feat",//45
            "fmtx",//46
            "fvar",//47
             //---------------

            "gvar",//48
            "hsty",//49
            "just",//50
            "lcar",//51
            "mort",//52
            "morx",//53
            "opbd",//54
            "prop",//55
            "trak",//56
            "Zapf",//57
            "Silf",//58
            "Glat",//59
            "Gloc",//60
            "Feat",//61
            "Sill",//62
            "...." //63 arbitrary tag follows
        };



        static bool ReadUIntBase128(BinaryReader reader, out uint result)
        {

            //UIntBase128 Data Type

            //UIntBase128 is a different variable length encoding of unsigned integers,
            //suitable for values up to 2^(32) - 1.

            //A UIntBase128 encoded number is a sequence of bytes for which the most significant bit
            //is set for all but the last byte,
            //and clear for the last byte.

            //The number itself is base 128 encoded in the lower 7 bits of each byte.
            //Thus, a decoding procedure for a UIntBase128 is: 
            //start with value = 0.
            //Consume a byte, setting value = old value times 128 + (byte bitwise - and 127).
            //Repeat last step until the most significant bit of byte is false.

            //UIntBase128 encoding format allows a possibility of sub-optimal encoding,
            //where e.g.the same numerical value can be represented with variable number of bytes(utilizing leading 'zeros').
            //For example, the value 63 could be encoded as either one byte 0x3F or two(or more) bytes: [0x80, 0x3f].
            //An encoder must not allow this to happen and must produce shortest possible encoding. 
            //A decoder MUST reject the font file if it encounters a UintBase128 - encoded value with leading zeros(a value that starts with the byte 0x80),
            //if UintBase128 - encoded sequence is longer than 5 bytes,
            //or if a UintBase128 - encoded value exceeds 232 - 1.

            //The "C-like" pseudo - code describing how to read the UIntBase128 format is presented below:
            //bool ReadUIntBase128(data, * result)
            //            {
            //                UInt32 accum = 0;

            //                for (i = 0; i < 5; i++)
            //                {
            //                    UInt8 data_byte = data.getNextUInt8();

            //                    // No leading 0's
            //                    if (i == 0 && data_byte = 0x80) return false;

            //                    // If any of top 7 bits are set then << 7 would overflow
            //                    if (accum & 0xFE000000) return false;

            //                    *accum = (accum << 7) | (data_byte & 0x7F);

            //                    // Spin until most significant bit of data byte is false
            //                    if ((data_byte & 0x80) == 0)
            //                    {
            //                        *result = accum;
            //                        return true;
            //                    }
            //                }
            //                // UIntBase128 sequence exceeds 5 bytes
            //                return false;
            //            }

            uint accum = 0;
            result = 0;
            for (int i = 0; i < 5; ++i)
            {
                byte data_byte = reader.ReadByte();
                // No leading 0's
                if (i == 0 && data_byte == 0x80) return false;

                // If any of top 7 bits are set then << 7 would overflow
                if ((accum & 0xFE000000) != 0) return false;
                //
                accum = (uint)(accum << 7) | (uint)(data_byte & 0x7F);
                // Spin until most significant bit of data byte is false
                if ((data_byte & 0x80) == 0)
                {
                    result = accum;
                    return true;
                }
                //
            }
            // UIntBase128 sequence exceeds 5 bytes
            return false;
        }

    }

    class Woff2Utils
    {

        const byte ONE_MORE_BYTE_CODE1 = 255;
        const byte ONE_MORE_BYTE_CODE2 = 254;
        const byte WORD_CODE = 253;
        const byte LOWEST_UCODE = 253;

        public static short[] ReadInt16Array(BinaryReader reader, int count)
        {
            short[] arr = new short[count];
            for (int i = 0; i < count; ++i)
            {
                arr[i] = reader.ReadInt16();
            }
            return arr;
        }
        public static ushort Read255UInt16(BinaryReader reader)
        {
            //255UInt16 Variable-length encoding of a 16-bit unsigned integer for optimized intermediate font data storage.
            //255UInt16 Data Type
            //255UInt16 is a variable-length encoding of an unsigned integer 
            //in the range 0 to 65535 inclusive.
            //This data type is intended to be used as intermediate representation of various font values,
            //which are typically expressed as UInt16 but represent relatively small values.
            //Depending on the encoded value, the length of the data field may be one to three bytes,
            //where the value of the first byte either represents the small value itself or is treated as a code that defines the format of the additional byte(s).
            //The "C-like" pseudo-code describing how to read the 255UInt16 format is presented below:
            //   Read255UShort(data )
            //    {
            //                UInt8 code;
            //                UInt16 value, value2;

            //                const oneMoreByteCode1    = 255;
            //                const oneMoreByteCode2    = 254;
            //                const wordCode            = 253;
            //                const lowestUCode         = 253;

            //                code = data.getNextUInt8();
            //                if (code == wordCode)
            //                {
            //                    /* Read two more bytes and concatenate them to form UInt16 value*/
            //                    value = data.getNextUInt8();
            //                    value <<= 8;
            //                    value &= 0xff00;
            //                    value2 = data.getNextUInt8();
            //                    value |= value2 & 0x00ff;
            //                }
            //                else if (code == oneMoreByteCode1)
            //                {
            //                    value = data.getNextUInt8();
            //                    value = (value + lowestUCode);
            //                }
            //                else if (code == oneMoreByteCode2)
            //                {
            //                    value = data.getNextUInt8();
            //                    value = (value + lowestUCode * 2);
            //                }
            //                else
            //                {
            //                    value = code;
            //                }
            //                return value;
            //            } 
            //Note that the encoding is not unique.For example, 
            //the value 506 can be encoded as [255, 253], [254, 0], and[253, 1, 250]. 
            //An encoder may produce any of these, and a decoder MUST accept them all.An encoder should choose shorter encodings,
            //and must be consistent in choice of encoding for the same value, as this will tend to compress better.



            byte code = reader.ReadByte();
            if (code == WORD_CODE)
            {
                /* Read two more bytes and concatenate them to form UInt16 value*/
                //int value = (reader.ReadByte() << 8) & 0xff00;
                //int value2 = reader.ReadByte();
                //return (ushort)(value | (value2 & 0xff));
                int value = reader.ReadByte();
                value <<= 8;
                value &= 0xff00;
                int value2 = reader.ReadByte();
                value |= value2 & 0x00ff;

                return (ushort)value;
            }
            else if (code == ONE_MORE_BYTE_CODE1)
            {
                return (ushort)(reader.ReadByte() + LOWEST_UCODE);
            }
            else if (code == ONE_MORE_BYTE_CODE2)
            {
                return (ushort)(reader.ReadByte() + (LOWEST_UCODE * 2));
            }
            else
            {
                return code;
            }
        }
    }
}
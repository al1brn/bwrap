#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:09:36 2021

@author: alain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:57:37 2021

@author: alain
"""

import os
import numpy as np

from ..breader import DataType, Struct, BFileReader, Flags
from ..vertparts import VertParts
from ..maths.closed_faces import closed_faces
from .textformat import CharFormat, TextFormat
from .glyphe import Glyphe


INT8    =  DataType.INT8
INT16   =  DataType.INT16
INT32   =  DataType.INT32
INT64   =  DataType.INT64

UINT8   = DataType.UINT8
UINT16  = DataType.UINT16
UINT32  = DataType.UINT32

FIXED   = DataType.FIXED
SFRAC   = DataType.SFRAC

# =============================================================================================================================
# A ttf table
    
class MainTable(Struct):
    def __init__(self, ttf, name, defs):

        self.ttf    = ttf
        self.name   = name

        self.table_offset = ttf.table_offset(name)
        self.ttf.reader.offset = self.table_offset

        super().__init__(self.ttf.reader, defs)
        
        self.loaded = True
        self.status = ""
        
    def __repr__(self):
        return f"'{self.name}' --> " + super().__repr__()
        
    
# =============================================================================================================================
# cmap tables

# -----------------------------------------------------------------------------------------------------------------------------
# cmap format 0
    
class CMap0(Struct):
    
    DEFS = {
        "length"         : UINT16,
        "language"       : UINT16,
        "glyphIndexArray": [256, UINT8]
        }
    
    def __init__(self, reader):
        super().__init__(reader, CMap0.DEFS)
        self.total_chars = 0
        
    def __repr__(self):
        return f"<CMap  0 ({self.length:6d}), lang[{self.language}], array: {self.glyphIndexArray[:10]}...>"
        
    def get_glyf_index(self, code):
        if code < 256:
            return self.glyphIndexArray[code]
        else:
            return 0
    
# -----------------------------------------------------------------------------------------------------------------------------
# cmap format 4
    
class CMap4(Struct):
    
    DEFS = {
        "length":         UINT16,
        "language":       UINT16,
        "segCount":      {"dtype": UINT16, "function": lambda x: x//2},
        "searchRange":   UINT16,
        "entrySelector": UINT16,
        "rangeShift":    UINT16,
        "endCode":       lambda reader: reader.read([reader.current[-1].segCount, UINT16]),
        "reservedPad":   UINT16,
        "startCode":     lambda reader: reader.read([reader.current[-1].segCount, UINT16]),
        "idDelta":       lambda reader: reader.read([reader.current[-1].segCount, INT16]),
        "idRangeOffset": lambda reader: reader.read([reader.current[-1].segCount + 1024, UINT16]), # Arbitrary cache value :-(
        #"glyphIndexArray": [1024, UINT16], 
        }
    
    def __init__(self, reader):
        super().__init__(reader, CMap4.DEFS)
        
    def __repr__(self):
        return f"<CMap  4 ({self.length:6d}), lang[{self.language}], segCount: {self.segCount}>"
    
    @property
    def total_chars(self):
        count = 0
        for iseg in range(len(self.startCode)):
            count += 1 + self.endCode[iseg] - self.startCode[iseg]
        return count
        
        
    def get_glyf_index(self, code):
        
        """
        If the idRangeOffset value for the segment is not 0, the mapping of character codes
        relies on glyphIdArray. The character code offset from startCode is added to the
        idRangeOffset value. This sum is used as an offset from the current location within
        idRangeOffset itself to index out the correct glyphIdArray value.
        
        This obscure indexing trick works because glyphIdArray immediately follows idRangeOffset
        in the font file. The C expression that yields the glyph index is:

            glyphId = *(idRangeOffset[i]/2
                        + (c - startCode[i])
                        + &idRangeOffset[i])

        The value c is the character code in question, and i is the segment index in which c appears.
        If the value obtained from the indexing operation is not 0 (which indicates missingGlyph), 
        idDelta[i] is added to it to get the glyph index. 
        
        The idDelta arithmetic is modulo 65536.
        
        If the idRangeOffset is 0, the idDelta value is added directly to the character code offset 
        (i.e. idDelta[i] + c) to get the corresponding glyph index. Again, the idDelta arithmetic
        is modulo 65536.
        
        """
        for i, c in enumerate(self.endCode):
            if code <= c:
                if code >= self.startCode[i]:
                    
                    idRangeOffset = self.idRangeOffset[i]

                    if idRangeOffset == 0:
                        return self.idDelta[i] + code
                    else:
                        ofs = idRangeOffset//2 + (code - self.startCode[i])
                        return self.idRangeOffset[i + ofs]
                else:
                    break
        
        return 0
     
    
# -----------------------------------------------------------------------------------------------------------------------------
# cmap format 12
    
class CMap12(Struct):
    
    DEFS = {
        "reserved":       UINT16,
        "length":         UINT32,
        "language":       UINT32,
        "nGroups":        UINT32,
        }
    
    GROUP = {
        "startCharCode":     UINT32,
        "endCharCode":       UINT32,
        "startGlyphCode":    UINT32,
        }

    
    def __init__(self, reader):
        super().__init__(reader, CMap12.DEFS)
        
        self.groups = reader.read_table(self.nGroups, CMap12.GROUP)
        
    def __repr__(self):
        return f"<CMap 12 ({self.length:6d}), lang[{self.language}], nGroups: {self.nGroups}>"
    
    @property
    def total_chars(self):
        count = 0
        for group in self.groups:
            count += 1 + group.endCharCode - group.startCharCode
        return count
    
        
    def get_glyf_index(self, code):
        for group in self.groups:
            if code >= group.startCharCode and code <= group.endCharCode:
                return group.startGlyphCode
        return 0

    
# -----------------------------------------------------------------------------------------------------------------------------
# cmap table
    
class CMap(MainTable):
    
    DEFS = {
        "version":         UINT16,
        "numberSubtables": UINT16
        }
    
    SUB_CMAP = {
        "platformID":         UINT16,
        "platformSpecificID": UINT16,
        "offset":             UINT32
        }
    
    def __init__(self, ttf):
        
        super().__init__(ttf, "cmap", CMap.DEFS)
        
        reader = self.ttf.reader
        
        cmaps = reader.read_table(self.numberSubtables, CMap.SUB_CMAP)
        self.cmaps = []
        
        for cm in cmaps:
            
            reader.offset = self.table_offset + cm.offset
            fmt = reader.read(INT16)
                
            if fmt == 0:
                cmap = CMap0(reader)
                
            elif fmt == 4:
                cmap = CMap4(reader)
                
            elif fmt == 12:
                cmap = CMap12(reader)
                
            else:
                self.loaded = False
                self.status = (f"cmap initialization error for {self.numberSubtables} cmaps: Unsupported cmap format: {fmt} {fmt == 0} {type(fmt)}")
                return
                
            cmap.platformID         = cm.platformID
            cmap.platformSpecificID = cm.platformSpecificID
            cmap.offset             = cm.offset
            cmap.format             = fmt
                
            self.cmaps.append(cmap)
                
    def __repr__(self):
        return f"<CMap ({self.numberSubtables}): {[cmap.format for cmap in self.cmaps]}>"
    
    def get_glyf_index(self, code):
        for cm in self.cmaps:
            code = cm.get_glyf_index(code)
            if code != 0:
                return code
        return 0

# =============================================================================================================================
# loca table

class Loca(MainTable):
    
    def __init__(self, ttf):
        
        super().__init__(ttf, "loca", defs={})
        
        reader = self.ttf.reader
        
        self.short_version = ttf.head.indexToLocFormat == 0
        
        # Read an addition record to compute the lengths
        if self.short_version:
            self.offsets = np.array(reader.read_array(ttf.maxp.numGlyphs + 1, UINT16))*2
        else:
            self.offsets = np.array(reader.read_array(ttf.maxp.numGlyphs + 1, UINT32))
            
        self.total = 0
        for ol in self:
            self.total += ol[1]
            
    def __repr__(self):
        s = f"'loca' --> <short: {self.short_version}, count={len(self.offsets)} from {min(self.offsets)} to {max(self.offsets)}\n"
        test = {i: self[i] for i in range(10)}
        s += f"{test}>"
        return s
    
    def __len__(self):
        # Take the additional record into account
        return len(self.offsets) - 1
    
    def __getitem__(self, index):
        if index > len(self):
            raise RuntimeError(f"TrueType {self.ttf.file_name}: index error in loca table. Index={index}, offset length={len(self)}")
        return (self.offsets[index], self.offsets[index+1] - self.offsets[index])
    

# =============================================================================================================================
# htmx table

class Hmtx(MainTable):
    
    ITEM = {
        "advanceWidth"    : UINT16,
        "leftSideBearing" : INT16
        }
    
    def __init__(self, ttf):
        
        super().__init__(ttf, "hmtx", defs={})
        
        reader = self.ttf.reader
        
        
        hm_count = ttf.hhea.numberOfHMetrics
        gl_count = ttf.maxp.numGlyphs
        
        count = max(hm_count, gl_count)
        self.hMetrics = np.zeros((count, 2), int)

        hMetrics  = reader.read_table(hm_count, Hmtx.ITEM)
        
        self.hMetrics[:hm_count, 0] = [hm.advanceWidth for hm in hMetrics]
        self.hMetrics[:hm_count, 1] = [hm.leftSideBearing for hm in hMetrics]

        if gl_count > hm_count:
            
            remain = gl_count - hm_count
            if remain > 100000:
                self.loaded = False
                self.status = f"HTMX load error. glyfs count: {gl_count} hMetric: {hm_count} remain: {remain}"
                return
            
            lsb = reader.read_array(remain, INT16)
            self.hMetrics[hm_count:, 0] = hMetrics[-1].advanceWidth
            self.hMetrics[hm_count:, 1] = lsb
            
    def __repr__(self):
        s = f"'htmx' --> <count: {len(self)}, \n"
        test = {i: self[i] for i in range(10)}
        s += f"{test}>"
        return s
    
    def __len__(self):
        return len(self.hMetrics)
    
    def __getitem__(self, index):
        if index > len(self):
            raise RuntimeError(f"TrueType {self.ttf.file_name}: index error in htmx table. Index={index}, offset length={len(self)}")
        return self.hMetrics[index]

# =============================================================================================================================
# name table

class Name(MainTable):
    
    D_ITEM = {
        "platformID"         : UINT16,
        "platformSpecificID" : UINT16,
        "languageID"         : UINT16,
        "nameID"             : UINT16,
        "length"             : UINT16,
        "offset"             : UINT16,
        }
    
    D_NAME = {
        "format"       : UINT16,
        "count"        : UINT16,
        "stringOffset" : UINT16,
        "nameRecord"   : lambda reader: reader.read_table(reader.current[-1].count, Name.D_ITEM) 
        }
    
    def __init__(self, ttf):
        
        super().__init__(ttf, "name", defs=Name.D_NAME)
        
        self.ids = {}
        for i in range(self.count):
            self.ids[self.nameRecord[i].nameID] = self[i]
        
    def __repr__(self):
        
        s = f"<Name: format={self.format}, count={self.count} offset={self.stringOffset} \n"
        for k in self.ids.keys():
            s += f"{k:6d}: {self.ids[k]}\n"
            
        return s + ">"
    
    def identifier(self, name_id):
        s = self.ids.get(name_id)
        if s is None:
            return "Undefined"
        else:
            return s
    
    def __len__(self):
        return len(self.nameRecord)
    
    def __getitem__(self, index):
        record = self.nameRecord[index]
        reader = self.ttf.reader
        reader.push()
        reader.open()
        reader.offset = self.ttf.table_offset('name') + self.stringOffset + record.offset
        sb = self.ttf.reader.read_bytes(record.length)
        reader.close()
        reader.pop()
        
        s = ""
        for c in sb:
            s += chr(c)
        
        return s
    

# =============================================================================================================================
# glyf

class Glyf(Struct):
    
    DEFS = {
        "numberOfContours"  : INT16,
        "xMin"              : INT16,
        "yMin"              : INT16,
        "xMax"              : INT16,
        "yMax"              : INT16,
        }
    
    SIMPLE = {
        "endPtsOfContours"  : lambda reader: reader.read([reader.current[-1].numberOfContours, UINT16]),
        "instructionLength" : UINT16,
        "instructions"      : lambda reader: reader.read([reader.current[-1].instructionLength, UINT8]),
        }

    COMPOUND = {
        "flags"      : Flags(UINT16, ['ARG_1_AND_2_ARE_WORDS', 'ARGS_ARE_XY_VALUES', 'ROUND_XY_TO_GRID','WE_HAVE_A_SCALE',
                                      None, 'MORE_COMPONENTS', 'WE_HAVE_AN_X_AND_Y_SCALE', 'WE_HAVE_A_TWO_BY_TWO', 'WE_HAVE_INSTRUCTIONS',
                                      'USE_MY_METRICS', 'OVERLAP_COMPOUND']),
        "glyphIndex" : UINT16,
        }

    
    def __init__(self, ttf, offset, glyf_index=None, code=None):
        
        #self.glyf_offset = offset
        self.ttf = ttf
        reader = self.ttf.reader
        
        reader.open()
        reader.offset = offset
        
        super().__init__(reader, Glyf.DEFS)
        self.glyf_index = glyf_index
        self.code       = code
        
        self.simple = self.numberOfContours >= 0
        
        def read_compound():
            
            comp = Struct(reader, Glyf.COMPOUND)
            
            if comp.ARG_1_AND_2_ARE_WORDS:
                if comp.ARGS_ARE_XY_VALUES:
                    atype = INT16
                else:
                    atype = UINT16
            else:
                if comp.ARGS_ARE_XY_VALUES:
                    atype = INT8
                else:
                    atype = UINT8
                    
            comp.arg1 = reader.read(atype)
            comp.arg2 = reader.read(atype)
            
            if comp.WE_HAVE_A_SCALE:
                comp.scale = reader.read(SFRAC)
                
            elif comp.WE_HAVE_AN_X_AND_Y_SCALE:
                comp.xscale = reader.read(SFRAC)
                comp.yscale = reader.read(SFRAC)
                
            elif comp.WE_HAVE_A_TWO_BY_TWO:
                comp.xscale  = reader.read(SFRAC)
                comp.scale01 = reader.read(SFRAC)
                comp.scale10 = reader.read(SFRAC)
                comp.yscale  = reader.read(SFRAC)
            
            return comp
        
        # ---------------------------------------------------------------------------
        # Simple glyphe
        
        if self.simple:
            
            reader.read_struct(self, Glyf.SIMPLE)
            
            # ----- Read the x & y coordinates
            
            def read_x(flags):
                if flags & 0x02:
                    x = reader.read(UINT8)
                    if (flags & 0x10) == 0:
                        x = -x
                else:
                    if flags & 0x10:
                        x = 0
                    else:
                        x = reader.read(INT16)
                return x

            def read_y(flags):
                if flags & 0x04:
                    y = reader.read(UINT8)
                    if (flags & 0x20) == 0:
                        y = -y
                else:
                    if flags & 0x20:
                        y = 0
                    else:
                        y = reader.read(INT16)
                return y
            
            def s_flags(flags):
                s = ""
                s += "x" if flags & 0x02 else "X"
                s += "y" if flags & 0x04 else "Y"
                s += "o" if flags & 0x08 else "-"
                s += "1" if flags & 0x10 else "0"
                s += "1" if flags & 0x20 else "0"
                if flags & 0x01: s= "[" + s + "]"
                return s
            
            # ---- Points to read
            
            points_count = self.endPtsOfContours[-1]+1
            self.flags = []
            self.xCoordinates = []
            self.yCoordinates = []

            # ---- Read the flags
            
            for i in range(points_count):
                
                flags = reader.read(UINT8)
                self.flags.append(flags)
                
                if flags & 0x08:
                    rept = reader.read(UINT8)
                    for i in range(rept):
                        self.flags.append(flags)
                        
                if len(self.flags) >= points_count:
                    break
                
            # ----- Read the x then y coordinates
            
            for i in range(points_count):
                self.xCoordinates.append(read_x(self.flags[i]))
                
            for i in range(points_count):
                self.yCoordinates.append(read_y(self.flags[i]))
                
            self.glyphe = Glyphe(self.ttf, self)
            self.loaded = True
            
        # ---------------------------------------------------------------------------
        # Compound glyphe
            
        else:
            self.glyphe = Glyphe(self.ttf)
            
            reader.push()
            for wd in range(100):

                reader.pop()
                comp = read_compound()
                reader.push()
                
                glyf = self.ttf.get_glyf(comp.glyphIndex, code=self.code)
                
                # ---------------------------------------------------------------------------
                # https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6glyf.html
                
                a = 1.
                b = 0.
                c = 0.
                d = 1.
                if comp.WE_HAVE_A_SCALE:
                    a = comp.scale
                    d = comp.scale
                elif comp.WE_HAVE_AN_X_AND_Y_SCALE:
                    a = comp.xscale
                    d = comp.yscale
                elif comp.WE_HAVE_A_TWO_BY_TWO:
                    a = comp.xscale
                    b = comp.scale01
                    c = comp.scale10
                    d = comp.yscale
                    
                if comp.ARGS_ARE_XY_VALUES:
                    e = comp.arg1
                    f = comp.arg2
                else:
                    e = 0.
                    f = 0.
                    
                m = max(abs(a), abs(b))
                if abs(abs(a)-abs(c)) < 33/65536:
                    m *= 2

                n = max(abs(c), abs(d))
                if abs(abs(b)-abs(d)) < 33/65536:
                    n *= 2
                    
                gl = glyf.glyphe.clone()
                pts = np.array(gl.points)
                gl.points[:, 0] = (m*(a/m*pts[:, 0] + c/m*pts[:, 1] + e)).astype(int)
                gl.points[:, 1] = (n*(b/n*pts[:, 0] + d/n*pts[:, 1] + f)).astype(int)
                
                self.glyphe.add_glyphe(gl)
                
                if not comp.MORE_COMPONENTS:
                    break
                
            reader.pop()
                
                
        self.glyphe.compute_contours()
        self.glyphe.glyf_index = self.glyf_index
        
        reader.close()
        
    # For debug: python source code
    def py_source(self):
        tab  = "    "
        tab2 = tab*2
        print(tab2 + f"glyf.glyf_index = {self.glyf_index}")
        print(tab2 + f"glyf.flags = {self.flags}")
        print(tab2 + f"glyf.xCoordinates = {self.xCoordinates}")
        print(tab2 + f"glyf.yCoordinates = {self.yCoordinates}")
        print(tab2 + f"glyf.endPtsOfContours = {self.endPtsOfContours}")
        print()
            
    def __len__(self):
        return len(self.contours)
    
    def __getitem__(self, index):
        return np.array(self.contours[index])
    
    @property
    def width(self):
        return self.hmtx[self.glyf_index][0]
 
    @property
    def lsb(self):
        return self.hmtx[self.glyf_index][1]
    
    @property
    def ascent(self):
        return self.yMax
    
    @property
    def descent(self):
        return self.yMin
    
    @property
    def height(self):
        return self.yMax - self.yMin
        
    def shape(self, index):
        i0 = 0 if index == 0 else self.endPtsOfContours[index-1]+1
        i1 = self.endPtsOfContours[index]+1
        return self.points[i0:i1, :]
    

        
# =============================================================================================================================
# Tables definitions

D_MAXP = {
    "version"               : FIXED,
    "numGlyphs"             : UINT16,
    "maxPoints"             : UINT16,
    "maxContours"           : UINT16,
    "maxComponentPoints"    : UINT16,
    "maxComponentContours"  : UINT16,
    "maxZones"              : UINT16,
    "maxTwilightPoints"     : UINT16,
    "maxStorage"            : UINT16,
    "maxFunctionDefs"       : UINT16,
    "maxInstructionDefs"    : UINT16,
    "maxStackElements"      : UINT16,
    "maxSizeOfInstructions" : UINT16,
    "maxComponentElements"  : UINT16,
    "maxComponentDepth"     : UINT16,
    }
    
D_HEAD = {
    "version"               : FIXED,
    "fontRevision"          : FIXED,
    "checkSumAdjustment"    : UINT32,
    "magicNumber"           : UINT32,
    "flags"                 : UINT16,
    "unitsPerEm"            : UINT16,
    "created"               : INT64,
    "modified"              : INT64,
    "xMin"                  : INT16,
    "yMin"                  : INT16,
    "xMax"                  : INT16,
    "yMax"                  : INT16,
    "macStyle"              : UINT16,
    "lowestRecPPEM"         : UINT16,
    "fontDirectionHint"     : UINT16,
    "indexToLocFormat"      : UINT16,
    "glyphDataFormat"       : UINT16,
    }

D_HHEA = {
    "majorVersion"          : UINT16,
    "minorVersion"          : UINT16,
    "ascender"              : INT16,
    "descender"             : INT16,
    "lineGap"               : INT16,
    "advanceWidthMax"       : UINT16,
    "minLeftSideBearing"    : INT16,
    "minRightSideBearing"   : INT16,
    "xMaxExtent"            : INT16,
    "caretSlopeRise"        : INT16,
    "caretSlopeRun"         : INT16,
    "caretOffset"           : INT16,
    "reserved"              : [4, INT16],
    "metricDataFormat"      : INT16,
    "numberOfHMetrics"      : UINT16,
    }

D_VHEA = {
    "version"               : UINT16,
    "ascent"                : INT16,
    "descent"               : INT16,
    "lineGap"               : INT16,
    "advanceHeightMax"      : INT16,
    "minTopSideBearing"     : INT16,
    "minBottomSideBearing"  : INT16,
    "yMaxExtent"            : INT16,
    "caretSlopeRise"        : INT16,
    "caretSlopeRun"         : INT16,
    "caretOffset"           : INT16,
    "reserved"              : [4, INT16],
    "metricDataFormat"      : INT16,
    "numOfLongVerMetrics"   : UINT16,        
    }

D_OS_2 = {
    "version"               : UINT16,
    "xAvgCharWidth"         : INT16,
    "usWeightClass"         : UINT16,
    "usWidthClass"          : UINT16,
    "fsType"                : UINT16,
    "ySubscriptXSize"       : INT16,
    "ySubscriptYSize"       : INT16,
    "ySubscriptXOffset"     : INT16,
    "ySubscriptYOffset"     : INT16,
    "ySuperscriptXSize"     : INT16,
    "ySuperscriptYSize"     : INT16,
    "ySuperscriptXOffset"   : INT16,
    "ySuperscriptYOffset"   : INT16,
    "yStrikeoutSize"        : INT16,
    "yStrikeoutPosition"    : INT16,
    "sFamilyClass"          : INT16,
    "panose"                : [10, UINT8],
    "ulUnicodeRange"        : [4, UINT32],
    "achVendID"             : [4, UINT8],
    "fsSelection"           : UINT16,
    "usFirstCharIndex"      : UINT16,
    "usLastCharIndex"       : UINT16,
    "sTypoAscender"         : INT16,
    "sTypoDescender"        : INT16,
    "sTypoLineGap"          : INT16,
    "usWinAscent"           : UINT16,
    "usWinDescent"          : UINT16,
}

D_OS_2_V1 = {
    "ulCodePageRange"       : [2, UINT32],
}

D_OS_2_V4 = {
    "ulCodePageRange"       : [2, UINT32],

    "sxHeight"              : INT16,
    "sCapHeight"            : INT16,
    "usDefaultChar"         : UINT16,
    "usBreakChar"           : UINT16,
    "usMaxContext"          : UINT16,
}

D_OS_2_V5 = {
    "ulCodePageRange"       : [2, UINT32],

    "sxHeight"              : INT16,
    "sCapHeight"            : INT16,
    "usDefaultChar"         : UINT16,
    "usBreakChar"           : UINT16,
    "usMaxContext"          : UINT16,
    
    "usLowerOpticalPointSize" : UINT16,
    "usUpperOpticalPointSize" : UINT16,    
}


# =============================================================================================================================
# The whole file      
#
# https://docs.microsoft.com/en-us/typography/opentype/spec/loca
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6loca.html
      
            
class Ttf(Struct):
    
    DEFS = {
        "Version"       : FIXED,
        "numTables"     : UINT16,
        "searchRange"   : UINT16,
        "entrySelector" : UINT16,
        "rangeShift"    : UINT16,
        }
    
    D_TABLE = {
        "name"          : "4",
        "check"         : UINT32,
        "offset"        : UINT32,
        "length"        : UINT32
        }
    
    def __init__(self, reader):
        
        self.reader = reader
        self.base_offset = reader.offset
        
        super().__init__(reader, Ttf.DEFS)
        self.loaded = False
        self.status = "Not initialized"
        self.font   = None  # Not yet loaded
        
        # ---------------------------------------------------------------------------
        # The tables
        
        tb = reader.read_table(self.numTables, Ttf.D_TABLE)
        self.tables = {}
        for t in tb:
            self.tables[t.name] = t
        
        # ----- Unsupported
        
        mandatory = ['name', 'loca', 'head', 'hhea', 'OS/2', 'maxp', 'hmtx']
        missing = []
        for mand in mandatory:
            if mand not in self.tables.keys():
                missing.append(mand)
                
        if len(missing) > 0:
            self.status = f"Missing mandatory tables: {missing}"
            return
        
        # ---------------------------------------------------------------------------
        # ----- name
        
        self.name = Name(self)
        
        # ---------------------------------------------------------------------------
        # ----- head
        
        self.head = MainTable(self, "head", D_HEAD)
        
        # ---------------------------------------------------------------------------
        # ----- hhea
        
        self.hhea = MainTable(self, "hhea", D_HHEA)

        # ---------------------------------------------------------------------------
        # ----- vhea
        
        if "vhea" in self.tables.keys():
            self.vhea = MainTable(self, "vhea", D_VHEA)
            
        # ---------------------------------------------------------------------------
        # ----- OS/2
        
        self.os_2 = MainTable(self, "OS/2", D_OS_2)
        if self.os_2.version == 5:
            reader.read_struct(self.os_2, defs=D_OS_2_V5)
        elif self.os_2.version == 4:
            reader.read_struct(self.os_2, defs=D_OS_2_V4)
        elif self.os_2.version > 0:
            reader.read_struct(self.os_2, defs=D_OS_2_V1)
            
        # ---------------------------------------------------------------------------
        # ----- Dimensions
        
        self.maxp = MainTable(self, "maxp", D_MAXP)
            
        # ---------------------------------------------------------------------------
        # ----- Locations
        
        self.loca = Loca(self)
        if not self.loca.loaded:
            self.loaded = False
            self.status = self.loca.status

        # ---------------------------------------------------------------------------
        # ----- hmtx
        
        self.hmtx = Hmtx(self)
        if not self.hmtx:
            self.loaded = False
            self.status = self.hmtx.status
            
        # ---------------------------------------------------------------------------
        # ----- the characters map
        
        self.cmap = CMap(self)
        if not self.cmap.loaded:
            self.loaded = False
            self.status = self.cmap.status
            
        # ---------------------------------------------------------------------------
        # ----- The default rasterization parameters
        
        self.raster_delta = 10
        self.raster_low   = True
        
        # ---------------------------------------------------------------------------
        # ----- Plane to display the chars
        
        self.display_plane = 'XY'
        
        # ---------------------------------------------------------------------------
        # ----- Load the default char
        
        default = Glyf(self, self.glyf_index_to_glyf_offset(0))
        self.glyfs   = {0: default}
        self.glyphes = {0: default.glyphe}
        self.empty_glyphe = Glyphe(self)
        
        # ---------------------------------------------------------------------------
        # ----- Load the main chars
        
        for code in range(32, 127):
            self.get_glyphe(code)
            
        # Default ratio for the end user metrics
        self.base_ratio   = 1/1560
        self.bold_base    = self.get_glyphe(ord('.')).width()*.3
        
        # ---------------------------------------------------------------------------
        # ----- Base for metrics
        
        self.base_space_width = self.get_glyphe(32).xwidth()
        
        h0 = self.hhea.lineGap
        h1 = self.get_glyphe(ord("M")).ascent() * 1.5
        self.base_line_height = max(h0, h1)
        
        # ---------------------------------------------------------------------------
        # ----- Loaded = ok
        
        self.loaded = True
        self.font   = self
        self.status = "Loaded"
        
        
    def table_offset(self, table_name):
        if True or table_name in ['cmap']:
            return self.tables[table_name].offset
        else:
            return self.base_offset + self.tables[table_name].offset
        
    @property
    def file_name(self):
        return self.reader.file_name
    
    @property
    def font_family(self):
        return self.name.identifier(1)
    
    @property
    def font_sub_family(self):
        return self.name.identifier(2)
    
    @property
    def identification(self):
        return self.name.identifier(3)
    
    @property
    def full_name(self):
        return self.name.identifier(4)
    
    @property
    def count(self):
        return self.maxp.numGlyphs
    
    def __repr__(self):
        if self.loaded:
            return f"<TTF font: {self.font_family} - {self.font_sub_family} ({self.count} glyphes)>"
        else:
            return f"<TTF font: UNLOADED: {self.status}>"
    
    def code_to_glyf_index(self, code):
        return self.cmap.get_glyf_index(code)
    
    def glyf_index_to_glyf_offset(self, glyf_index):
        """Read the loca table to get the offset of a glyf index.
        
        Return None for empty glyfs"""
        
        ofs, length = self.loca[glyf_index]
        if length == 0:
            return self.table_offset('glyf') if ofs == 0 else None
        else:
            return self.table_offset('glyf') + ofs
    
    def code_to_glyf_offset(self, code):
        return self.glyf_index_to_glyf_offset(self.code_to_glyf_index(code))
    
    def get_glyf(self, glyf_index, code):
        
        # ----- Is the glyf already in the cache
        
        glyf = self.glyfs.get(glyf_index)
        if glyf is not None:
            return glyf
        
        # ----- Get the glyf offset
        glyf_offset = self.glyf_index_to_glyf_offset(glyf_index)
        if glyf_offset is None:
            return None
        
        # ----- The glyf exists, let's load it
        
        glyf = Glyf(self, glyf_offset, glyf_index=glyf_index, code=code)

        self.glyfs[glyf_index] = glyf
        return glyf
    
    def get_glyphe(self, code):
        
        # ----- Check if the glyphe is in the cache
        glyphe = self.glyphes.get(code)
        if glyphe is not None:
            return glyphe
        
        # ----- Glyf index from cmap
        glyf_index = self.code_to_glyf_index(code)
        if glyf_index == 0:
            return self.get_glyphe(0)
        
        # ----- Check if glyf exists from loca
        glyf_offset, length = self.loca[glyf_index]
        if length == 0:
            glyphe = Glyphe(self)
            glyphe.glyf_index  = glyf_index
            glyphe.code        = code
            self.glyphes[code] = glyphe
            return glyphe
        
        # ----- Load the glyf
        glyf = self.get_glyf(glyf_index, code)

        self.glyphes[code] = glyf.glyphe
        return glyf.glyphe
    
    # ===========================================================================
    # Metrics interface
    
    def ratio(self, text_format):
        if text_format is None:
            return self.base_ratio
        else:
            return self.base_ratio * text_format.scale
    
    # space width without interchars
    def raw_space_width(self, text_format):
        if text_format is None:
            return self.base_space_width * self.base_ratio
        else:
            return self.base_space_width * self.base_ratio * text_format.scale
        #return self.base_space_width * self.ratio * (1. + self.interchars)
    
    # space width with interchars
    def space_width(self, text_format):
        if text_format is None:
            return self.base_space_width * self.base_ratio
        else:
            return self.base_space_width * self.base_ratio * text_format.scale * (1. + text_format.interchars)
        #return self.base_space_width * self.ratio * (1. + self.interchars)
    
    def line_height(self, text_format):
        if text_format is None:
            return self.base_line_height * self.base_ratio
        else:
            return self.base_line_height * self.base_ratio * text_format.scale * (1. + text_format.interlines)
        #return self.base_line_height * self.ratio
    
    def adapt_char_format(self, char_format, text_format=None):
        
        cf = CharFormat.Copy(char_format)
        cf.bold_shift = int(self.bold_base*cf.bold)
        cf.scale = cf.scales(self.ratio(text_format))
        
        if text_format is not None:
            cf.x_base = text_format.char_x_base
        
        return cf
    
    def char_metrics(self, c, char_format=None, text_format=None):
        
        glyphe = self.get_glyphe(ord(c))
        
        class Metrics():
            pass
        
        # Adapt the char format (bold_shift and scale)
        
        cf = self.adapt_char_format(char_format, text_format)
        rx = cf.xscale
        ry = cf.yscale
        #yscale = 1 if char_format is None else char_format.yscale
        
        m = Metrics()
        
        m.xMin     = glyphe.xMin(char_format) * rx
        m.xMax     = glyphe.xMax(char_format) * rx
        m.xwidth   = glyphe.xwidth(char_format) * rx
        m.lsb      = glyphe.lsb(char_format) * rx
        m.width    = m.xMax - m.xMin

        # After is the difference between xwidth and width plus a fraction of space
        # proportionnaly to the interchars parameter
        x_space = 0 if text_format is None else self.raw_space_width(text_format)*text_format.interchars 
        m.after    = m.xwidth - m.width + x_space

        m.descent  = glyphe.yMin(char_format) * ry
        m.ascent   = glyphe.yMax(char_format) * ry
        m.height   = m.ascent - m.descent
        
        m.line_height = self.line_height(text_format)
        
        return m
    
    # ===========================================================================
    # Access to the curves
    
    def mesh_char(self, c, char_format=None, text_format=None, return_faces=False):
        
        cf = self.adapt_char_format(char_format, text_format)
        
        return self.get_glyphe(ord(c)).raster(
            delta           = self.raster_delta, 
            lowest_geometry = self.raster_low,
            char_format     = cf, 
            plane           = self.display_plane, 
            return_faces    = return_faces)
    
    # ---------------------------------------------------------------------------
    # Curve : return the array of beziers points
    
    def beziers_char(self, c, char_format=None, text_format=None):

        cf = self.adapt_char_format(char_format, text_format)
        
        return self.get_glyphe(ord(c)).beziers(char_format=cf, plane='XY')
        

    # ---------------------------------------------------------------------------
    # Curve : return the stack of bezier points and the profile
    
    def curve_char(self, c, char_format=None, text_format=None):
        
        beziers = self.beziers_char(c, char_format, text_format)

        verts   = np.zeros((0, 3), np.float)
        profile = np.zeros((len(beziers), 3), int)
        for i, bz in enumerate(beziers):
            verts = np.append(verts, bz[:, 0], axis=0)
            verts = np.append(verts, bz[:, 1], axis=0)
            verts = np.append(verts, bz[:, 2], axis=0)
            profile[i] = (3, len(bz), 0)
            
        return verts, profile
    
    # ---------------------------------------------------------------------------
    # Build a VertParts geometry
    
    def geometry(self, c, char_format=None, text_format=None, type='mesh'):
        
        if type=='Mesh':
            verts, faces, uvs = self.mesh_char(c, char_format, text_format, return_faces=True)
            geo = VertParts.MeshFromData(verts, faces, uvs)
        else:
            verts, profile = self.curve_char(c, char_format, text_format)
            geo = VertParts.CurveFromData(verts, profile)
            
        return geo
    
    # ---------------------------------------------------------------------------
    # Update the geometry to take formatting in account
    
    def update_geometry(self, geometry, part, c, char_format=None, text_format=None):
        if geometry.type == 'Mesh':
            geometry.set_verts(self.mesh_char(c, char_format, text_format, return_faces=False), part=part)
        else:
            geometry.set_verts(self.curve_char(c, char_format, text_format)[0], part=part)
    

# =============================================================================================================================
# Collection file
#
# https://docs.fileformat.com/font/ttc/
            
class Ttc(Struct):
    
    DEFS = {
        "tag"           : "4",
        "numTables"     : UINT16,
        "majorVersion"  : UINT16,
        "minorVersion " : UINT16,
        "numFonts"      : UINT16,
        "offsets"       : lambda reader: reader.read_array(reader.current[-1].numFonts, UINT32)
        }
    
    D_TTF = {
        "Version"       : FIXED,
        "numTables"     : UINT16,
        "searchRange"   : UINT16,
        "entrySelector" : UINT16,
        "rangeShift"    : UINT16,
        "tables"        : lambda reader: reader.read_table(reader.current[-1].numTables, Ttf.D_TABLE)
        }
    
    
    def __init__(self, reader, headers_only=False):
        
        self.reader = reader

        super().__init__(reader, Ttc.DEFS)
        
        self.fonts = []
        for offset in self.offsets:
            
            reader.offset = offset
            
            if headers_only:
                self.fonts.append(Struct(reader, Ttc.D_TTF))
            else:
                font = Ttf(reader)
                if font.loaded:
                    self.fonts.append(font)
                    
        self.font_index = 0
            
    def __repr__(self):
        s = "<Font collection\n"
        for font in self.fonts:
            s += f"     {font}\n"
        return s + ">"
    
    def __len__(self):
        return len(self.fonts)
    
    def __getitem__(self, index):
        return self.fonts[index]
    
    @property
    def font(self):
        if len(self.fonts) > 0:
            return self[self.font_index]
        else:
            return None

# =============================================================================================================================
# Read font (either ttf or ttc)

class Font():
    def __init__(self, file_name):

        self.reader = BFileReader(file_name)
        
        fname, ftype = os.path.splitext(file_name)
        self.fname = fname
        
        self.collection = ftype.lower() == '.ttc'
        
        if self.collection:
            self.ttc = Ttc(self.reader)
        else:
            self.ttf = Ttf(self.reader)
            
        self.reader.close()
        
            
    @property
    def loaded(self):
        if self.collection:
            return self.ttc.font is not None
        else:
            return self.ttf.loaded
        
    def __len__(self):
        if self.loaded:
            if self.collection:
                return len(self.ttc)
            else:
                return 1
        else:
            return 0
        
    def __getitem__(self, index):
        index = np.clip(index, 0, len(self)-1)
        if self.collection:
            return self.ttc[index]
        else:
            return self.ttf
        
    # ----------------------------------------------------------------------------------------------------
    # Current font
        
    @property
    def font_index(self):
        if self.collection:
            return self.ttc.font_index
        else:
            return 0
        
    @font_index.setter
    def font_index(self, value):
        if value >= len(self):
            s = f"Invalid font index: {value}. The number of fonts in '{self.fname}' is {len(self)}"
            raise RuntimeError(s)
            
        if self.collection:
            self.ttc.font_index = value            
        
    @property
    def currrent_font(self):
        return self[self.font_index]
    
    # ----------------------------------------------------------------------------------------------------
    # Current font
        
    @property
    def family(self):
        return [font.font_family for font in self]
        
    @property
    def sub_family(self):
        return [font.font_sub_family for font in self]
    
    # ===========================================================================
    # Plane
    
    @property
    def display_plane(self):
        return self[0].display_plane
    
    @display_plane.setter
    def display_plane(self, value):
        if value.upper() == 'XZ':
            plane = 'XZ'
        elif value.upper() == 'YZ':
            plane = 'YZ'
        else:
            plane = 'XY'
        for i in range(len(self)):
            self[i].display_plane = plane
    
    # ===========================================================================
    # Attributes
    
    @property
    def raster_delta(self):
        return self[0].raster_delta
    
    @raster_delta.setter
    def raster_delta(self, value):
        for i in range(len(self)):
            self[i].raster_delta = value
    
    @property
    def raster_low(self):
        return self[0].raster_low
    
    @raster_low.setter
    def raster_low(self, value):
        for i in range(len(self)):
            self[i].raster_low = value    
    
    # ===========================================================================
    # Metrics interface
    
    @property
    def scale(self):
        return self[0].scale
    
    @scale.setter
    def scale(self, value):
        for i in range(len(self)):
            self[i].scale = value

    @property
    def interchars(self):
        return self[0].interchars
    
    @interchars.setter
    def interchars(self, value):
        for i in range(len(self)):
            self[i].interchars = value
    
    @property
    def space_width(self):
        return self[0].space_width
    
    def char_metrics(self, c, char_format=None, text_format=None):
        index = 0 if char_format is None else char_format.font
        return self[index].char_metrics(c, char_format, text_format)
    
    

folder = "/System/Library/Fonts/Supplemental/"
file_name = "Arial.ttf"    
file_name = "Georgia.ttf"    
file_name = "Herculanum.ttf"
file_name = "Krungthep.ttf"
file_name = "Zapfino.ttf"
file_name = "Tahoma.ttf"
file_name = "Apple Chancery.ttf"

font = Font(folder + file_name)

ttf = font.currrent_font

#ttf = Ttf(BFileReader(folder + file_name))

gl = ttf.get_glyphe(ord("é"))
#gl.plot_raster()






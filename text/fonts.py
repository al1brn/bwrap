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

from ..maths.closed_faces import closed_faces

INT8    =  0
INT16   =  1
INT32   =  2
INT64   =  3

UINT8   = 10
UINT16  = 11
UINT32  = 12

FIXED   = 20
SFRAC   = 21


def defs_str(obj, defs):
    s = ""
    sepa = ""
    for k in defs:
        s += sepa + f"{k}: {getattr(obj, k)}"
        sepa = ", "
    return s

# -----------------------------------------------------------------------------------------------------------------------------
# Base structure being read 

class Struct():
    def __init__(self, reader, defs):
        
        reader.read_struct(self, defs)
        self._lock = 0
        
    def __repr__(self):
        
        if self._lock > 0:
            return "<self>"
        
        self._lock += 1
            
        s = f"<Struct {type(self).__name__}:\n"
        for k in dir(self):
            if (k[0] != '_') and ((k.upper() != k) or (len(k) >= 8)) and k not in ['ttf']:
                v = getattr(self, k)
                sv = f"{v}"
                if len(sv) > 30:
                    sv = sv[:30] + " ..."
                    if hasattr(v, '__len__'):
                        sv += f" ({len(v)})"
                s += f"     {k:25s}: {sv}\n"
                
        self._lock -= 1
                
        return s + ">"
    
# =============================================================================================================================
# Flags reading

class Flags():
    def __init__(self, dtype, names):
        self.dtype = dtype
        self.names = names
        
    def read(self, reader, struct):
        
        flags = reader.read(self.dtype)
        
        bit = 1
        for i, nm in enumerate(self.names):
            if nm is not None:
                setattr(struct, nm, flags & bit > 0)
            bit = bit << 1
            
        return flags
    

# =============================================================================================================================
# The binary reader

# -----------------------------------------------------------------------------------------------------------------------------
# The base class and the node class

class BReader():
    def __init__(self, owner=None, base_offset=0, name="Binary reader"):
        
        self.owner       = owner
        self.base_offset = base_offset
        self.name        = name
        
        self.breaders    = {}
        self.current     = []
        
        self.opens       = 0
        self.stack       = []
        
        self.open()

    # =============================================================================================================================
    # Nodes management
        
    @property
    def top(self):
        return self if self.owner is None else self.owner
    
    def new_reader(self, name, base_offset):
        self.breaders[name] = BReader(self, base_offset, name)
        return self.breaders[name]
    
    @property
    def file_name(self):
        return self.top.file_name
    
    # =============================================================================================================================
    # Basics
        
    def open(self):
        self.opens += 1
        self.owner.open()
        self.offset = 0
        
    def close(self):
        
        if self.opens == 0:
            raise RuntimeError("BReader close error: impossible to close a closed reader")

        self.opens -= 1
        self.owner.close()
        
    @property
    def offset(self):
        return self.owner.offset - self.base_offset
    
    @offset.setter
    def offset(self, value):
        self.owner.offset = self.base_offset + value
        
    def read_bytes(self, count):
        return self.top.file.read(count)
        
    def push(self):
        self.stack.append(self.offset)
        
    def pop(self):
        if len(self.stack) == 0:
            raise RuntimeError("BReader pop error: impossible to pop an empty stack.")
        self.offset = self.stack.pop()
        
    # =============================================================================================================================
    # Common methods
        
    @staticmethod
    def to_int(bts):
        r = 0
        for b in bts:
            r = (r << 8) + b
        return r
    
    def read(self, dtype):

        if dtype == UINT8:
            return self.to_int(self.read_bytes(1))
        elif dtype == UINT16:
            return self.to_int(self.read_bytes(2))
        elif dtype == UINT32:
            return self.to_int(self.read_bytes(4))
        
        if dtype == INT8:
            i = self.to_int(self.read_bytes(1))
            if i > 0x7F:
                return i - 0x100
            else:
                return i
        elif dtype == INT16:
            i = self.to_int(self.read_bytes(2))
            if i > 0x7FFF:
                return i - 0x10000
            else:
                return i
        elif dtype == INT32:
            i = self.to_int(self.read_bytes(4))
            if i > 0x7FFFFFFF:
                return i - 0x100000000
            else:
                return i
        elif dtype == INT64:
            return self.to_int(self.read_bytes(8))
        
        elif issubclass(type(dtype), Flags):
            return dtype.read(self, self.current[-1])
        
        elif dtype == FIXED:
            return self.read(INT32)/65536.
        elif dtype == SFRAC:
            return self.read(INT16)/0x4000
        
        elif type(dtype) is list:
            return [self.read(dtype[1]) for i in range(dtype[0])]
        
        elif type(dtype) is dict:
            return dtype["function"](self.read(dtype["dtype"]))
        
        elif type(dtype).__name__ == 'function':
            return dtype(self)

        elif type(dtype) is str:
            bts = self.read_bytes(int(dtype))
            s = ""
            for b in bts:
                s += chr(b)
            return s

        else:
            raise RuntimeError(f"Unknwon data type {dtype}")
        
    def read_struct(self, obj, defs=None, offset=None):

        if offset is not None:
            self.offset = offset
            
        if defs is None:
            defs = obj._defs

        self.current.append(obj)        
        for k in defs:
            setattr(obj, k, self.read(defs[k]))
        self.current.pop()
            
        return obj
    
    def read_table(self, count, defs):

        table = []
        for i in range(count):
            table.append(Struct(self, defs))
            
        return table
    
    def read_array(self, count, dtype):

        table = []
        for i in range(count):
            table.append(self.read(dtype))
            
        return table        

# -----------------------------------------------------------------------------------------------------------------------------
# Top class which actually read bytes
       
class BFileReader(BReader):
    
    def __init__(self, file_name):
        self.file_name_ = file_name
        self.file = None
        super().__init__()
        
    def open(self):
        self.opens += 1
        if self.opens == 1:
            self.file = open(self.file_name, "br")
        self.offset = 0
        
    def close(self):
        if self.opens == 0:
            raise RuntimeError("BReader close error: impossible to close a closed reader")
        self.opens -= 1
        if self.opens == 0:
            self.file.close()
            self.file = None
            
    @property
    def file_name(self):
        return self.file_name_
        
    @property
    def offset(self):
        return self.file.tell()
    
    @offset.setter
    def offset(self, value):
        self.file.seek(value)
    
        

    
# =============================================================================================================================
# cmap table
    
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
# cmap table

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
    
    def dump(self):
        print('-'*60)
        print("Dump de cmap0")
        print()
        print(f"{self.glyphIndexArray[:30]}...")
        print()
    
    
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
                        if False:
                            print("segments", len(self.endCode))
                            for iseg in range(len(self.startCode)):
                                for ic in range(self.startCode[iseg], self.endCode[iseg]+1):
                                    ro = self.idRangeOffset[iseg]
                                    s = f"{ic:4d}: seg: {iseg:3d}, idRangeOffset {ro:4d}"
                                    if ro == 0:
                                        iglyf = self.idDelta[iseg] + ic
                                    else:
                                        ofs = ro // 2 + (ic - self.startCode[iseg])
                                        s += f", diff: {(ic - self.startCode[iseg])}, ofs: {ofs}"
                                        iglyf = self.idRangeOffset[iseg + ofs]
                                    s += f" --> {iglyf}"
                                    print(s)
    
                            raise RuntimeError("idRangeOffset not yet supported")
                        
                        ofs = idRangeOffset//2 + (code - self.startCode[i])
                        #print("search for", code, len(self.endCode))
                        #print(self.startCode[:20])
                        #print(self.endCode[:20])
                        #print("idRangeOffset", idRangeOffset, (code - self.startCode[i]), ofs, "segment", i, "-->", self.idRangeOffset[i + ofs])
                        return self.idRangeOffset[i + ofs]
                    
                        raise RuntimeError("idRangeOffset not yet supported")
                else:
                    break
        
        return 0
    
    def dump(self):
        print('-'*60)
        print("Dump de cmap4")
        print()
        print(f"{len(self.endCode)} segments to map {self.total_chars} chars")
        for iseg in range(len(self.startCode)):
            print(f" > segment {iseg}: [{self.startCode[iseg]:3d} - {self.endCode[iseg]:3d}]")
            wd = 0
            for ic in range(self.startCode[iseg], self.endCode[iseg]+1):
                ro = self.idRangeOffset[iseg]
                cc = " " if ic < 32 else chr(ic)
                s = f"     {ic:4d} ({cc}): idRangeOffset= {ro:4d}"
                if ro == 0:
                    iglyf = self.idDelta[iseg] + ic
                else:
                    ofs = ro // 2 + (ic - self.startCode[iseg])
                    s += f", diff= {(ic - self.startCode[iseg])}, ofs= {ofs}"
                    if iseg + ofs > len(self.idRangeOffset):
                        s += f" OUT OF RANGE {iseg + ofs} > {len(self.idRangeOffset)}"
                        iglyf = 0
                    else:
                        iglyf = self.idRangeOffset[iseg + ofs]
                s += f" --> {iglyf:5d}"
                print(s)
                
                wd += 1
                if wd == 5:
                    print("...\n")
                    break
            
            print()
            if iseg == 5:
                break
        
    
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
    
    def dump(self):
        print('-'*60)
        print("Dump de cmap12")
        print()
        print(f"{len(self.groups)} groups to map {self.total_chars} chars")
        for ig, group in enumerate(self.groups):
            print(f" > group {ig:3d}: [{group.startCharCode:3d} - {group.endCharCode}]")
            wd = 0
            for ic in range(group.startCharCode, group.endCharCode+1):
                cc = " " if ic < 32 else chr(ic)
                print(f"     {ic:4d} ({cc}) --> {group.startGlyphCode:5d}")
                wd += 1
                if wd == 5:
                    print("     ...\n")
                    break
            print()
            if ig > 5:
                break

    
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
        
    
    def dump(self):
        print("="*60)
        print(f"cmap dump: {len(self.cmaps)} cmaps {[cm.format for cm in self.cmaps]} ({self.numberSubtables})")
        print()

        for cm in self.cmaps:
            cm.dump()
                
        if len(self.cmaps) > 1:
            oks = 0
            kos = 0
            wd = 0
            for i in range(0, 256):
                codes = [cm.get_glyf_index(i) for cm in self.cmaps]
                oks += 1
                for v in codes:
                    if v != codes[0]:
                        kos += 1
                        oks -= 1
                        if wd <= 20:
                            print(f"   KO: {i:3d}", codes)
                            if wd == 20:
                                print("...")
                        wd += 1
                        break
            print()
            print(f"Several cmaps formats: oks={oks} kos={kos}")
            print()

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
        
        #print(f"hm_count {hm_count} gl_count {gl_count} shape {self.hMetrics.shape}")
        
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
# glyf table

# -----------------------------------------------------------------------------------------------------------------------------
# A raw glyphe (which can be imported in a compound glyf)

class Glyphe():
    
    def __init__(self, ttf, glyf=None):
        
        self.ttf        = ttf
        
        self.on_curve   = []
        self.points     = None
        self.ends       = []
        self.xMin       = 0
        self.yMin       = 0
        self.xMax       = 0
        self.yMax       = 0
        self.contours   = []
        self.beziers_   = None
        self.glyf_index = None
        
        if glyf is not None:
            self.add(glyf)
            
    def __repr__(self):
        s = f"<Glyphe of {len(self.points)} points"
        if self.empty:
            s += " empty glyphe"
        else:
            s += f" --> {len(self)} contours"
        return s + ">"
    
    def clone(self):
        clone = Glyphe(self.ttf)
        clone.on_curve = list(self.on_curve)
        clone.points   = np.array(self.points)
        clone.ends     = list(self.ends)
        return clone
    
    def check(self):
        if len(self.points) > 1000:
            raise RuntimeError(f"The number of points loojs weird {len(self.points)}")
            
    
    def add(self, glyf):
        
        self.on_curve.extend([(flag & 0x01) > 0 for flag in glyf.flags])

        pts = np.stack((np.cumsum(glyf.xCoordinates), np.cumsum(glyf.yCoordinates)), axis=1)
        if self.points is None:
            count = 0
            self.points = pts
        else:
            count = len(self.points)
            self.points = np.append(self.points, pts, axis=0)
        
        self.ends.extend([count + i for i in glyf.endPtsOfContours])
        
        self.xMin = min(self.points[:, 0])
        self.yMin = min(self.points[:, 1])
        self.xMax = max(self.points[:, 0])
        self.yMax = max(self.points[:, 1])
        
        self.check()

    def add_glyphe(self, glyphe):
        
        self.on_curve.extend(glyphe.on_curve)
        
        if self.points is None:
            count = 0
            self.points = np.array(glyphe.points)
        else:
            count = len(self.points)
            self.points = np.append(self.points, glyphe.points, axis=0)
        
        self.ends.extend([count + i for i in glyphe.ends])
        
        self.xMin = min(self.points[:, 0])
        self.yMin = min(self.points[:, 1])
        self.xMax = max(self.points[:, 0])
        self.yMax = max(self.points[:, 1])
        
        self.check()
        
    def compute_contours(self):
        
        self.contours = []
        contour = None
        for index in range(len(self.points)):
            
            # Current point
            pt  = np.array(self.points[index])
            
            # ----- First of the contour
            if contour is None:
                contour   = np.array([pt])
                last_OC   = self.on_curve[index]
                OC0       = last_OC
            
            else:
                # Add an intermediary point to altern on and off curve points
                if self.on_curve[index] == last_OC:
                    contour = np.append(contour, [(contour[-1] + pt)/2], axis=0)
                    
                # Add the point
                contour = np.append(contour, [pt], axis=0)
                last_OC = self.on_curve[index]
                
            # ----- Last point of the curve
            if index in self.ends:
                
                # Intermediary point with the first one 
                if last_OC == OC0:
                    contour = np.append(contour, [(contour[0] + pt)/2], axis=0)
                    
                # Ensure flag 0 is on curve
                if not OC0:
                    cont = np.array(contour)
                    contour[1:] = cont[:-1]
                    contour[-1] = cont[0]
                    
                # Let's start a new contour
                self.contours.append(contour.reshape(len(contour)>>1, 2, 2))
                contour = None
                
    def __len__(self):
        return len(self.contours)
                
    def __getitem__(self, index):
        return np.array(self.contours[index])
    
    @property
    def empty(self):
        return len(self.contours) == 0
    
    @property
    def xwidth(self):
        if self.glyf_index is None:
            return self.xMax - self.xMin
        else:
            return self.ttf.hmtx[self.glyf_index][0]

    @property
    def width(self):
        return self.xMax-self.xMin
    
    @property
    def after(self):
        return self.xwidth - self.width

    @property
    def lsb(self):
        if self.glyf_index is None:
            return self.xMin
        else:
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
    
    # ===========================================================================
    # Blender Bezier vertices
    # Return a list or array(n, 3, 3):
    # - n : vertices count
    # - 3 : verts, lefts an rights is this order
    # - 3 : 3D vectors
    
    @property
    def beziers(self):
        
        if self.beziers_ is None:
            
            beziers = []
            for pts in self.contours:
                
                n = len(pts)
                
                bz = np.zeros((n, 3, 3), np.float)

                bz[:,   0, :2] = pts[:, 0]
                
                bz[1:,  1, :2] = pts[1:  , 0]*.3333 + pts[:-1, 1]*0.6667
                bz[:-1, 2, :2] = pts[ :-1, 0]*.3333 + pts[:-1, 1]*0.6667
                
                bz[ 0,  1, :2] = pts[ 0, 0]*.3333 + pts[-1, 1]*0.6667
                bz[-1,  2, :2] = pts[-1, 0]*.3333 + pts[-1, 1]*0.6667
                
                beziers.append(bz)
                
            self.beziers_ = beziers
            
        return self.beziers_
    
    # ===========================================================================
    # Rasterization
    # Points and faces
    
    def raster(self, delta=10):
        
        verts         = None
        closed_curves = []
        
        # ---------------------------------------------------------------------------
        # Compute the rasterized contours
        
        for contour in self.contours:
            
            # Short for len(contour)
            n = len(contour)

            #print('-'*20, "\nNew contour of length", n)
            
            # ----- Compute intermediary points at index based on the precision

            def comp_points(index, prec):
                
                t = np.expand_dims(np.linspace(0, 1, max(2, int(round(prec)))), axis=1)
                c = t*t
                b = 2*t*(1-t)
                a = (1-t)*(1-t)
        
                return a*contour[index%n, 0] + b*contour[index%n, 1] + c*contour[(index+1)%n, 0]
            
            # ----- Compute the n curves to form the final closed curve
            
            closed_curve = []
            first_point = 0 if verts is None else len(verts)
            for index in range(n):
                #print("    curve", index, "/", n-1)
                
                # Estimate the number of points for the curve
                # The lesser the cross product, the lesser the precision
                
                v0 = contour[(index+1)%n, 0] - contour[index, 0]
                v1 = contour[index, 1] - contour[index, 0]
                prec = np.sqrt(abs(v0[0]*v1[1] - v0[1]*v1[0])) / delta
                
                # Compute the points
                # No need of the last point which will be the first of the
                # following curve
                
                ras_verts = comp_points(index, prec)[:-1]
                
                # Add the computed points to the current series
                vert0 = 0 if verts is None else len(verts)
                
                if verts is None:
                    verts = ras_verts
                else:
                    verts = np.append(verts, ras_verts, axis=0)
                    
                # Extend the closed curve
                # Need one 
                
                closed_curve.extend([vert0 + i for i in range(len(ras_verts))])
                
                # If this is the last curve, the last index must loop to the first point of the curve
                #if index == n-1:
                #    closed_curve[-1] = first_point
                    
                #print("    verts", verts.shape, "closed_curve", closed_curve[:5], closed_curve[-5:])
                
            closed_curves.append(closed_curve)
            
        if False:
            print("vertices = np.array([")
            for v in verts:
                print(f"    [{v[0]}, {v[1]}],")
            print("])")
            print("faces = [")
            for c in closed_curves:
                print(f"    {c},")
            print("]")
            
        # ---------------------------------------------------------------------------
        # Compute the holed faces with the curve
            
        return np.insert(verts, 2, 0, axis=1), closed_faces(verts, closed_curves)
    
    # ===========================================================================
    # DEBUG on matplotlib
    
    def plot_raster(self, delta=10, show_points = False, **kwargs):
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.set(**kwargs)
        ax.set_aspect(1.)
        ax.plot((0, 1000, 1000, 0), (0, 0, 1000, 1000), '.')
        
        verts, faces = self.raster(delta)
        
        line = '.-k' if show_points else '-k'
        
        for face in faces:
            x = verts[[edge[0] for edge in face], 0]
            y = verts[[edge[0] for edge in face], 1]
            ax.plot(x, y, line)
        
        plt.show()        
    
    
    # ===========================================================================
    # DEBUG on matplotlib
    
    def plot_points(self, prec=12):
        
        t = np.expand_dims(np.linspace(0, 1, prec), axis=1)
        c = t*t
        b = 2*t*(1-t)
        a = (1-t)*(1-t)
        
        plots = []
        
        for pts in self:
            
            points = None
            
            n = len(pts)
            for k in range(n):
                
                cs = a*pts[k%n, 0] + b*pts[k%n, 1] + c*pts[(k+1)%n, 0]
                
                if points is None:
                    points = cs
                else:
                    points = np.append(points, cs, axis=0)
                    
            plots.append(points)
            
        return plots
        
    
    def plot_bezier(self, curve=True, **kwargs):
        
        import matplotlib.pyplot as plt
        
        prec = 12
        t = np.expand_dims(np.linspace(0, 1, prec), axis=1)
        
        d = t*t*t
        c = 3*t*t*(1-t)
        b = 3*t*(1-t)*(1-t)
        a = (1-t)*(1-t)*(1-t)
        
        fig, ax = plt.subplots()
        ax.set(**kwargs)
        ax.set_aspect(1.)
        
        beziers = self.beziers
        
        for bz in beziers:
            
            verts  = bz[:, 0]
            lefts  = bz[:, 1]
            rights = bz[:, 2]
            
            if curve:

                points = None
                
                for i in range(len(verts)):
                    
                    cs = a*verts[i] + b*rights[i] + c*lefts[(i+1)%len(verts)] + d*verts[(i+1)%len(verts)]
                    
                    if points is None:
                        points = cs
                    else:
                        points = np.append(points, cs, axis=0)
                        
                if points is not None:
                    ax.plot(points[:, 0], points[:, 1], '-k')
                    
                
            
            else:
                ax.plot(verts[:, 0], verts[:, 1],  'ok')
                ax.plot(rights[:, 0], rights[:, 1], '.b')
                ax.plot(lefts[:, 0], lefts[:, 1],  '.r')
        
        plt.show()    
    
    def plot(self, curve=True, **kwargs):
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        ax.set(**kwargs)
        ax.set_aspect(1.)
        
        if curve:
            base = self.plot_points()
            for points in base:
                ax.plot(points[:, 0], points[:, 1], '-k')
                
        else:
            for pts in self:
                ax.plot(pts[:, 0, 0], pts[:, 0, 1], '-')
                ax.plot(pts[:, 1, 0], pts[:, 1, 1], '.')
        
        plt.show()    


# -----------------------------------------------------------------------------------------------------------------------------
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

    
    def __init__(self, ttf, offset, glyf_index=None):
        
        #self.glyf_offset = offset
        self.ttf = ttf
        reader = self.ttf.reader
        
        reader.open()
        reader.offset = offset
        
        super().__init__(reader, Glyf.DEFS)
        self.glyf_index = glyf_index
        
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
                
            #print("flags", [s_flags(flag) for flag in self.flags])
                    
            # ----- Read the x then y coordinates
            
            for i in range(points_count):
                self.xCoordinates.append(read_x(self.flags[i]))
                
            for i in range(points_count):
                self.yCoordinates.append(read_y(self.flags[i]))
                
                
            # =========================
            
            self.glyphe = Glyphe(self.ttf, self)
                
            # ----- The raw points of the glyf
            self.raw_points = np.stack((np.cumsum(self.xCoordinates), np.cumsum(self.yCoordinates)), axis=1)
            
            # ----- Let's normalize the coordinates in a normalized square
            r = 1 / (ttf.os_2.sTypoAscender - ttf.os_2.sTypoDescender)
            self.points = self.raw_points * r
            self.xMin = self.xMin * r
            self.yMin = self.yMin * r
            self.xMax = self.xMax * r
            self.yMax = self.yMax * r
            
            # ---- Create the contours with the intermediary points
            
            self.contours = []
            contour = None
            last_flag = 0x00
            for index in range(len(self.points)):
                
                # Current point
                pt = np.array(self.points[index])
                
                # First of the contour
                if contour is None:
                    contour   = [pt]
                    last_flag = self.flags[index]
                    flag0     = last_flag
                    
                # Add an intermediary point to altern on and off curve points
                else:
                    if self.flags[index] & 0x01:
                        if last_flag & 0x01:
                            contour.append((contour[-1] + pt)/2)
                    else:
                        if (last_flag & 0x01) == 0:
                            contour.append((contour[-1] + pt)/2)
                    contour.append(pt)
                    last_flag = self.flags[index]
                    
                # Last point of the curve
                if index in self.endPtsOfContours:
                    
                    # Intermediary point with the first one 
                    if (last_flag & 0x01) == (flag0 & 0x01):
                        contour.append((contour[0] + pt)/2)
                        
                    # Ensure flag 0 is on curve
                    if (flag0 & 0x01) == 0:
                        cont = list(contour)
                        contour = [cont[1:]]
                        contour.append(cont[0])
                        
                    # Let's start a new contour
                    self.contours.append(contour)
                    contour = None
                    
            self.loaded = True
            
        else:
            self.glyphe = Glyphe(self.ttf)
            
            reader.push()
            for wd in range(100):

                reader.pop()
                comp = read_compound()
                reader.push()
                
                glyf = self.ttf.get_glyf(comp.glyphIndex)
                
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
                
                #print(glyf)
                if not comp.MORE_COMPONENTS:
                    break
                
            reader.pop()
                
                
        self.glyphe.compute_contours()
        self.glyphe.glyf_index = self.glyf_index
        
        reader.close()
                
            
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
        #print(f"Tables: {self.tables.keys()}")
        
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
        self.ratio        = 1/1560
        
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
    
    def get_glyf(self, glyf_index):
        
        # ----- Is the glyf already in the cache
        
        glyf = self.glyfs.get(glyf_index)
        if glyf is not None:
            return glyf
        
        # ----- Get the glyf offset
        glyf_offset = self.glyf_index_to_glyf_offset(glyf_index)
        if glyf_offset is None:
            return None
        
        # ----- The glyf exists, let's load it
        
        glyf = Glyf(self, glyf_offset, glyf_index=glyf_index)
        

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
            self.glyphes[code] = glyphe
            return glyphe
        
        # ----- Load the glyf
        glyf = self.get_glyf(glyf_index)

        self.glyphes[code] = glyf.glyphe
        return glyf.glyphe
    
    # ===========================================================================
    # Metrics interface
    
    @property
    def ratio(self):
        return self.ratio_
    
    @ratio.setter
    def ratio(self, value):
        self.ratio_ = value
        self.space_width = self.get_glyphe(32).xwidth * self.ratio_
        h = self.hhea.lineGap
        if h == 0:
            h = self.get_glyphe(ord("M")).ascent *1.5
        self.line_height = h * self.ratio_
    
    def char_xwidth(self, c):
        return self.get_glyphe(ord(c)).xwidth * self.ratio_

    def char_width(self, c):
        return self.get_glyphe(ord(c)).width * self.ratio_

    def char_after(self, c):
        return self.get_glyphe(ord(c)).after * self.ratio_
    
    def char_wa(self, c):
        gl = self.get_glyphe(ord(c))
        return gl.width * self.ratio_, gl.after * self.ratio_
    
    # ===========================================================================
    # Char_beziers
    
    def char_beziers(self, c):
        beziers = self.get_glyphe(ord(c)).beziers
        return [bz*self.ratio_ for bz in beziers]
    
    # ===========================================================================
    # Geometry
    
    @staticmethod
    def curved_shape(contour, prec=12):

        t = np.expand_dims(np.linspace(0, 1, prec), axis=1)
        c = t*t
        b = 2*t*(1-t)
        a = (1-t)*(1-t)
        
        pts    = np.reshape(contour, (len(contour)//2, 2, 2))
        points = None
        for k in range(len(pts)):
            
            cs = a*pts[k, 0, :] + b*pts[k, 1, :] + c*pts[(k+1) % len(pts), 0, :]
            
            if points is None:
                points = cs
            else:
                points = np.append(points, cs, axis=0)
        
        return points
    
    
    def get_curves(self, word):

        contours = []

        x = 0.
        for c in word:
            glyphe = self.get_glyphe(ord(c))
            for contour in glyphe:
                pts = np.array(contour, np.float)
                pts[:, 0] += x
                contours.append(pts)
            
            x += glyphe.width
                
        return contours
    
    def plot_word(self, word, **kwargs):
        
        import matplotlib.pyplot as plt
        
        if not self.loaded:
            return
            
        curves = self.get_curves(word)
        
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        
        for curve in curves:
            pts = self.curved_shape(curve)
            ax.plot(pts[:, 0], pts[:, 1], 'black')
            
        fname, ftype = os.path.splitext(self.file_name)
        ax.set(**kwargs)
        
        plt.show()
        
        
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
                    
        self.current_font = 0
            
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
            return self[self.current_font]
        else:
            return None

# =============================================================================================================================
# Read font (either ttf or ttc)

def load_font(file_name):
    
    fname, ftype = os.path.splitext(file_name)
        
    reader = BFileReader(file_name)

    if ftype.lower() == '.ttf':
        font = Ttf(reader)
    elif ftype.lower() == '.ttc':
        font = Ttc(reader)
        
    reader.close()
        
    if font.font is None:
        return None
    else:
        return font

    
# =============================================================================================================================
# Some tests

def tests():
    
    fonts_folder = "/System/Library/Fonts/"
    
    fname_test = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

    def read_font(file_name):
    
        reader = BFileReader(file_name)
        
        fname, ftype = os.path.splitext(file_name)
        fname = fname.split()[-1]
        if ftype.lower() == '.ttf':
            print("Open ttf:", fname)
            font = Ttf(reader)
        elif ftype.lower() == '.ttc':
            print("Open collection:", fname)
            font = Ttc(reader)
            
        reader.close()
            
        print("   Read ttc:", font)
        
        return font
        
        
    ZapfDingbats= "ZapfDingbats.ttf"
    NewYork= "NewYork.ttf"
    Arial = "Supplemental/Arial Unicode.ttf"
    Helvetica = "Helvetica.ttc"
    AppleSDGothicNeo = "AppleSDGothicNeo.ttc"
    HelveticaNeue = "HelveticaNeue.ttc"
    Noteworthy ="Noteworthy.ttc"
    MarkerFelt = "MarkerFelt.ttc"
    
    def test_font(file_name, all_coll = False):
        
        def plot_ttf(ttf, index=None):
            if ttf.loaded:
                label = f"{ttf.font_family} - {ttf.font_sub_family}"
                if index is not None:
                    label += f" ({index})"
                #ttf.plot_word("Cet été passé @ où", xlabel=label)
                #ttf.plot_word("ABC abc 12345", xlabel=label)
                c = "C"
                glyphe = ttf.get_glyphe(ord(c))
                glyphe.plot_bezier(True, title=ttf.font_family, xLabel="A")
                glyphe.plot_raster(delta=10, show_points=False, title=ttf.font_family, xLabel="A")
                #glyphe.plot(True, title=ttf.font_family, xLabel="A")
                
            else:
                print("     ", ttf.status)
        
        fname, ftype = os.path.splitext(file_name)
        
        if ftype.lower() == '.ttf':
            print("True type font:", fname)
            ttf = read_font(fonts_folder + file_name)
            plot_ttf(ttf)
            
        elif ftype.lower() == '.ttc':
            print("True type collection:", file_name)
            font = read_font(fonts_folder + file_name)
            if all_coll:
                for i, ttf in enumerate(font.fonts):
                    plot_ttf(ttf, i)
            else:
                plot_ttf(font.fonts[0], 0)
            
        
        
    
    def testall():
        
        dirList = os.listdir(fonts_folder) # current directory
        
        i = 0
        wd = 0
        for file_name in dirList:
            if not os.path.isdir(file_name):
                test_font(file_name, True)
    
    test_font(MarkerFelt, all_coll=False)  
    #testall()
    
        
#tests()

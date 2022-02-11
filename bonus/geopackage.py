#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:20:07 2022

@author: alain
"""

# Specifications from
# https://www.geopackage.org/spec131/index.html

import struct
import numpy as np
import sqlite3

from sqlite3 import Error

# ---------------------------------------------------------------------------
# An instance of a table row

class DBObject():
    def __init__(self, fields, row):
        self.fields = fields
        for n, v in zip(fields, row):
            setattr(self, n, v)
            
    def __repr__(self):
        s = ""
        for n in self.fields:
            s += f"   {n:15s}: {getattr(self, n)}\n"
        return s

# ---------------------------------------------------------------------------
# A table in the data base

class DBTable():
    def __init__(self, db, table_name):
        self.db         = db
        self.table_name = table_name
        self.fields     = self.db.get_table_cols(self.table_name)
        self.aliases    = {}
        self.objects    = None
        
    def __repr__(self):
        s = f"<Table {self.table_name} with {len(self.fields)} columns and {len(self)} rows:\n"
        if self.objects is None:
            for field in self.fields:
                s += f"   {field}\n"
        else:
            for i, obj in enumerate(self.objects):
                s += f"Row {i} -----\n{obj}\n"
                
        s += ">"
        return s
        
    def __len__(self):
        return self.db.sql(f"SELECT count(*) from {self.table_name}")[0][0]
    
    def set_alias(self, field, alias):
        self.aliases[field] = alias
        
    def fields_names(self):
        return [name if self.aliases.get(name) is None else self.aliases.get(name) for name in self.fields]
    
    def load_objects(self, equals={}):
        rows = self.db.read_table(self.table_name, fields=self.fields, fields_alias=self.aliases, equals=equals)
        fields = self.fields_names()
        self.objects = [DBObject(fields, row) for row in rows]
        
    def load_columns(self, fields, equals={}):
        
        count = len(fields)
        rows  = self.db.read_table(self.table_name, fields, equals=equals)
        size  = len(rows)
        vals = [[0]*size for i in range(count)]
        
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                vals[j][i] = v
                
        return [np.array(vs) for vs in vals]
    
    def read(self, fields, equals):
        return self.db.read_table(self.table_name, fields=fields, equals=equals)
    
    def read_value(self, field, equals):
        return self.db.read_value(self.table_name, field=field, equals=equals)
        
    
# ---------------------------------------------------------------------------
# A General DB reader

class DB():
    
    def __init__(self, fname):
        self.conn   = DB.connect(fname)
        self.tables = self.sql("SELECT name FROM sqlite_master WHERE type='table';")
        
    def __del__(self):
        print(f"Closing db: {self.conn}")
        if self.conn is not None:
            self.conn.close()
    
    @staticmethod
    def connect(fname):
        print(f"SQLITE: Connecting to db '{fname}'")
        conn = None
        try:
            conn = sqlite3.connect(fname)
            print(f"Version: {sqlite3.version}")
        except Error as e:
            print(e)
            
        return conn
    
    # ---------------------------------------------------------------------------
    # Columns of a table
    
    def get_table_cols(self, table_name):
        
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        return [descr[0] for descr in cur.description]
    
    # ---------------------------------------------------------------------------
    # Sql request
    
    def sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)

        rows = cur.fetchall()
        
        return rows
    
    # ---------------------------------------------------------------------------
    # Read a table
    
    def read_table(self, table_name, fields=None, fields_alias={}, equals={}):
        
        sql = "SELECT"
        if fields is None:
            sql += " *"
        else:
            sep = " "
            for field in fields:
                sql += sep + f"{field}"
                alias = fields_alias.get(field)
                if alias is not None:
                    sql += f" AS '{alias}'"
                sep = ", "
                
        sql += f" FROM {table_name}"
        
        if equals is not None:
            sep = " WHERE "
            for field, value in equals.items():
                sql += f"{sep}{field}={value}"
                sep = " AND "
                
        print("::: read table with sql: ", sql)
        
        return self.sql(sql)   
    
    # ---------------------------------------------------------------------------
    # Read a single value

    def read_value(self, table_name, field, equals={}):
        return self.read_table(table_name, fields=[field], equals=equals)[0][0]
    
# ---------------------------------------------------------------------------
# Geometry

class Geometry():
    def __init__(self, blob):
        self.blob = blob
        self.g_start = self.geometry_index
        
    def __repr__(self):
        s = "<Geometry:\n"
        s += f"header:        {self.blob[:8]}\n"
        s += f"magic:         {self.magic}\n"
        s += f"version:       {self.version}\n"
        s += f"flags:         {self.flags}\n"
        s += f" binary type:  {self.binary_type} {self.binary_type_name}\n"
        s += f" empty:        {self.empty} {self.empty_name}\n"
        s += f" envelop code: {self.envelop_code} {self.envelop_name}\n"
        s += f" byte_order:   {self.byte_order} {self.byte_order_name}\n"
        s += f"srs_id:        {self.srs_id}\n"
        s += f"{self.envelop}\n"
        s += f"Data length:   {len(self.blob) - self.g_start} bytes"
        
        return s
        
    @property
    def magic(self):
        return self.blob[:2]
    
    @property
    def version(self):
        return int(self.blob[2:3].hex(), base=16)
    
    @property
    def flags(self):
        return int(self.blob[3:4].hex(), base=16)
    
    @property
    def binary_type(self):
        return (self.flags & (1 << 5)) >> 5
    
    @property
    def binary_type_name(self):
        return ["StandardGeoPackageBinary", "ExtendedGeoPackageBinary"][self.binary_type]
    
    @property
    def empty(self):
        return (self.flags & (1 << 4)) >> 4

    @property
    def empty_name(self):
        return ["Not empty", "Empty"][self.binary_type]
    
    @property
    def envelop_code(self):
        return (self.flags & (7 << 1)) >> 1
    
    @property
    def envelop_name(self):
        return ["no envelop", 
                "[minx, maxx, miny, maxy], 32 bytes",
                "[minx, maxx, miny, maxy, minz, maxz], 48 bytes",
                "[minx, maxx, miny, maxy, minm, maxm], 48 bytes",
                "[minx, maxx, miny, maxy, minz, maxz, minm, maxm], 64 bytes",
                "Invalid", "Invalid", "Invalid", ][self.envelop_code]
    
    @property
    def envelop_size(self):
        return [0, 32, 48, 48, 64][self.envelop_code]
    
    @property
    def envelop_attrs(self):
        attrs = []
        if self.envelop_code >= 1:
            attrs.extend(["minx", "maxx", "miny", "maxy"])
        if self.envelop_code in [2, 4]:
            attrs.extend(["minz", "maxz"])
        if self.envelop_code in [3, 5]:
            attrs.extend(["minm", "maxm"])
        return attrs
    
    
    @property
    def byte_order(self):
        return self.flags & 1
    
    @property
    def byte_order_name(self):
        return ["Big Endian (most significant byte first)", "Little Endian (least significant byte first)"][self.byte_order]
    
    @property
    def big_indian(self):
        return self.byte_order == 0
    
    @property
    def little_indian(self):
        return self.byte_order == 1
    
    def read_int(self, index, size=4):
        rg = range(size)
        if self.big_indian:
            rg = range(size)
        else:
            rg = reversed(range(size))
        
        s = ""
        for i in rg:
            s += self.blob[index+i:index+i+1].hex()
        return int(s, base=16)
    
    def read_double(self, index, size=8):
        return struct.unpack('d', self.blob[index:index+size])[0]
    
    @property
    def srs_id(self):
        return self.read_int(4, size=4)
    
    @property
    def envelop(self):
        env = {}
        attrs = self.envelop_attrs
        for i, attr in enumerate(attrs):
            env[attr] = self.read_double(8+i*8, size=8)
            
        return env
    
    @property
    def geometry_index(self):
        return 8 + self.envelop_size
    
    
    
    
# ---------------------------------------------------------------------------
# A geo package reader

class GeoPack(DB):
    
    # Possible values for gpkg_geometry_columns.geometry_type_name
    GEO_TYPE = ['GEOMETRY','POINT','LINESTRING','POLYGON','MULTIPOINT','MULTILINESTRING','MULTIPOLYGON','GEOMETRYCOLLECTION']
    
    def __init__(self, fname):
        self.fname = fname
        super().__init__(fname)
        self.load()
        
    # ---------------------------------------------------------------------------
    # Content
    
    def __repr__(self):
        s = f"<GeoPackage {self.fname}:\n\n"
        
        s += "Spatial ref sys:\n"
        s += "----------------\n"
        for o in self.srs.objects:
            s += f"   {o.srs_id:4d}: {o.srs_name}\n"
        s += "\n"
            

        s += "Contents:\n"
        s += "---------\n"
        for c in self.contents.objects:
            rows = self.geocol.read(["column_name", "geometry_type_name"], equals={"table_name": f"'{c.table_name}'"})
            s += f"   {c.table_name:20s}: {c.data_type} {rows}\n"
            
            
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Load
    
    def load(self):
        
        self.srs = DBTable(self, "gpkg_spatial_ref_sys")
        self.srs.load_objects()
        
        self.contents = DBTable(self, "gpkg_contents")
        self.contents.load_objects()
        
        self.geocol = DBTable(self, "gpkg_geometry_columns")
        self.geocol.load_objects()
        
        return
        
        print("----------------")
        print("Spatial ref sys:")
        print("----------------")
        print()
        print(self.srs)

        print("---------")
        print("Contents:")
        print("---------")
        print()
        print(self.contents)

        print("-----------------")
        print("Geometry columns:")
        print("-----------------")
        print(self.geocol)
        
        
        
    
    
    # ---------------------------------------------------------------------------
    # print rows
    
    @staticmethod
    def print_rows(rows, title, fields=None):
        
        print()
        print(title)
        print("-"*len(title))
        
        if fields is not None:
            mlen = max([len(n) for n in fields])
        
        for i, row in enumerate(rows):
            if fields is None:
                print(row)
            else:
                for n, v in zip(fields, row):
                    print(f"   {n:10s}: {v}")
                print()
            if i > 20:
                break
            
        print()
    
    # ---------------------------------------------------------------------------
    # Tables
    
    def tables(self):
        tables = self.sql("SELECT name FROM sqlite_master WHERE type='table';")
        self.print_rows(tables, "DB tables")
        

geo = GeoPack(r"/Users/alain/Downloads/gadm36_FRA_gpkg/gadm36_FRA.gpkg")

print(geo)


tbf = DBTable(geo, "gadm36_FRA_0")
#tbf.load_objects()
print(tbf)
sql = "SELECT geom FROM gadm36_FRA_0"
rows = geo.sql(sql)
for row in rows:
    print()
    print('-'*30)
    g = Geometry(row[0])
    print(g)

"""
tbf = DBTable(geo, "gadm36_FRA_1")
#tbf.load_objects()
print(tbf)

for table in geo.tables:
    print(table)

#geo.tables()
"""


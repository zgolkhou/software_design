…о
≤6Г6
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ь
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
Є
AsString

input"T

output"
Ttype:
2		
"
	precisionint€€€€€€€€€"

scientificbool( "
shortestbool( "
widthint€€€€€€€€€"
fillstring 
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
°
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
…
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0ю€€€€€€€€"
value_indexint(0ю€€€€€€€€"+

vocab_sizeint€€€€€€€€€(0€€€€€€€€€"
	delimiterstring	И
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
TouttypeИ
2
LookupTableSizeV2
table_handle
size	И
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint€€€€€€€€€"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Р
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ј
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
®
SparseSegmentSqrtN	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
Љ
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
ј
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
StringSplit	
input
	delimiter
indices	

values	
shape	"

skip_emptybool(
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
;
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8ИФ

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
К
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
Й
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
:*
dtype0*.
value%B#BLanguageCodeBhourBmonth
x
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBB	TitleText
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
Ґ
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/Const*
Tdense
2*°
_output_shapesО
Л:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€::::€€€€€€€€€*
dense_shapes
:*

num_sparse*
ragged_split_types
 *
ragged_value_types
 *
sparse_types
2		
g
module/tokenPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
module/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"Ћџ А   
W
module/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
r
module/zerosFillmodule/zeros/shape_as_tensormodule/zeros/Const*
T0*!
_output_shapes
:ЋЈ;А
М
;module/embeddings/part_0/PartitionedInitializer/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        
Л
:module/embeddings/part_0/PartitionedInitializer/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ћџ А   
ю
5module/embeddings/part_0/PartitionedInitializer/SliceSlicemodule/zeros;module/embeddings/part_0/PartitionedInitializer/Slice/begin:module/embeddings/part_0/PartitionedInitializer/Slice/size*
Index0*
T0*!
_output_shapes
:ЋЈ;А
П
module/embeddings/part_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:ЋЈ;А*)
shared_namemodule/embeddings/part_0
Б
9module/embeddings/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpmodule/embeddings/part_0*
_output_shapes
: 
Њ
module/embeddings/part_0/AssignAssignVariableOpmodule/embeddings/part_05module/embeddings/part_0/PartitionedInitializer/Slice*+
_class!
loc:@module/embeddings/part_0*
dtype0
µ
,module/embeddings/part_0/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0*+
_class!
loc:@module/embeddings/part_0*!
_output_shapes
:ЋЈ;А*
dtype0
`
module/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
v
module/ExpandDims
ExpandDimsmodule/tokenmodule/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
j
)module/DenseToSparseTensor/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
Я
#module/DenseToSparseTensor/NotEqualNotEqualmodule/ExpandDims)module/DenseToSparseTensor/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
y
"module/DenseToSparseTensor/indicesWhere#module/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
®
!module/DenseToSparseTensor/valuesGatherNdmodule/ExpandDims"module/DenseToSparseTensor/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
w
&module/DenseToSparseTensor/dense_shapeShapemodule/ExpandDims*
T0*
_output_shapes
:*
out_type0	
W
module/tokenize/ConstConst*
_output_shapes
: *
dtype0*
value	B B 
Н
module/tokenize/StringSplitStringSplitmodule/tokenmodule/tokenize/Const*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:
a
 module/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB B 
°
.module/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsmodule/tokenize/StringSplitmodule/tokenize/StringSplit:1module/tokenize/StringSplit:2 module/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
}
module/IdentityIdentity.module/SparseFillEmptyRows/SparseFillEmptyRows*
T0	*'
_output_shapes
:€€€€€€€€€
}
module/Identity_1Identity0module/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*#
_output_shapes
:€€€€€€€€€
a
module/Identity_2Identitymodule/tokenize/StringSplit:2*
T0	*
_output_shapes
:
^
module/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
e

module/MaxMaxmodule/Identitymodule/Max/reduction_indices*
T0	*
_output_shapes
:
N
module/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
`
module/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
X
module/ones_like/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
m
module/ones_likeFillmodule/ones_like/Shapemodule/ones_like/Const*
T0	*
_output_shapes
:
T

module/AddAdd
module/Maxmodule/ones_like*
T0	*
_output_shapes
:
X
module/MaximumMaximummodule/Const
module/Add*
T0	*
_output_shapes
:
∞
!module/string_to_index/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*B
shared_name31module/hash_table_/tmp/tmpAmEwJw/tokens.txt_-2_-1*
value_dtype0	
r
'module/string_to_index/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
й
;module/string_to_index/hash_table/table_init/asset_filepathConst*
_output_shapes
: *
dtype0*~
valueuBs BmC:\Users\ALEXYA~1\AppData\Local\Temp\tfhub_modules\32f2b2259e1cc8ca58c876921748361283e73997\assets\tokens.txt
г
,module/string_to_index/hash_table/table_initInitializeTableFromTextFileV2!module/string_to_index/hash_table;module/string_to_index/hash_table/table_init/asset_filepath*
	key_indexю€€€€€€€€*
value_index€€€€€€€€€
П
)module/string_to_index_Lookup/hash_bucketStringToHashBucketFastmodule/Identity_1*#
_output_shapes
:€€€€€€€€€*
num_buckets§
я
/module/string_to_index_Lookup/hash_table_LookupLookupTableFindV2!module/string_to_index/hash_tablemodule/Identity_1'module/string_to_index/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:€€€€€€€€€
}
-module/string_to_index_Lookup/hash_table_SizeLookupTableSizeV2!module/string_to_index/hash_table*
_output_shapes
: 
∞
!module/string_to_index_Lookup/AddAdd)module/string_to_index_Lookup/hash_bucket-module/string_to_index_Lookup/hash_table_Size*
T0	*#
_output_shapes
:€€€€€€€€€
Ї
&module/string_to_index_Lookup/NotEqualNotEqual/module/string_to_index_Lookup/hash_table_Lookup'module/string_to_index/hash_table/Const*
T0	*#
_output_shapes
:€€€€€€€€€
—
module/string_to_index_LookupSelect&module/string_to_index_Lookup/NotEqual/module/string_to_index_Lookup/hash_table_Lookup!module/string_to_index_Lookup/Add*
T0	*#
_output_shapes
:€€€€€€€€€
Г
2module/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Е
4module/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Е
4module/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
в
,module/embedding_lookup_sparse/strided_sliceStridedSlicemodule/Identity2module/embedding_lookup_sparse/strided_slice/stack4module/embedding_lookup_sparse/strided_slice/stack_14module/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_mask
Ц
#module/embedding_lookup_sparse/CastCast,module/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
Л
%module/embedding_lookup_sparse/UniqueUniquemodule/string_to_index_Lookup*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Я
Cmodule/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0*!
_output_shapes
:ЋЈ;А*
dtype0
µ
8module/embedding_lookup_sparse/embedding_lookup/IdentityIdentityCmodule/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*
T0*!
_output_shapes
:ЋЈ;А
°
/module/embedding_lookup_sparse/embedding_lookupResourceGathermodule/embeddings/part_0%module/embedding_lookup_sparse/Unique*
Tindices0	*V
_classL
JHloc:@module/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
dtype0
В
:module/embedding_lookup_sparse/embedding_lookup/Identity_1Identity/module/embedding_lookup_sparse/embedding_lookup*
T0*V
_classL
JHloc:@module/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А
с
module/embedding_lookup_sparseSparseSegmentSqrtN:module/embedding_lookup_sparse/embedding_lookup/Identity_1'module/embedding_lookup_sparse/Unique:1#module/embedding_lookup_sparse/Cast*
T0*(
_output_shapes
:€€€€€€€€€А
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
t
save/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0*!
_output_shapes
:ЋЈ;А*
dtype0
_
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*!
_output_shapes
:ЋЈ;А
e
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*!
_output_shapes
:ЋЈ;А

save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*3
value*B(Bglobal_stepBmodule/embeddings
А
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*0
value'B%B B973771 128 0,973771:0,128
Э
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOpsave/Identity_1*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
С
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(Bglobal_stepBmodule/embeddings
Т
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*0
value'B%B B973771 128 0,973771:0,128
≠
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*%
_output_shapes
::ЋЈ;А*
dtypes
2	
N
save/Identity_2Identitysave/RestoreV2*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_2*
dtype0	
Y
save/Identity_3Identitysave/RestoreV2:1*
T0*!
_output_shapes
:ЋЈ;А
c
save/AssignVariableOp_1AssignVariableOpmodule/embeddings/part_0save/Identity_3*
dtype0
J
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1
Ё
checkpoint_initializer/prefixConst"/device:CPU:0*
_output_shapes
: *
dtype0*А
valuewBu BoC:\Users\ALEXYA~1\AppData\Local\Temp\tfhub_modules\32f2b2259e1cc8ca58c876921748361283e73997\variables\variables
Е
#checkpoint_initializer/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB
embeddings
Ш
'checkpoint_initializer/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*.
value%B#B973771 128 0,973771:0,128
”
checkpoint_initializer	RestoreV2checkpoint_initializer/prefix#checkpoint_initializer/tensor_names'checkpoint_initializer/shape_and_slices"/device:CPU:0*!
_output_shapes
:ЋЈ;А*
dtypes
2
X
IdentityIdentitycheckpoint_initializer*
T0*!
_output_shapes
:ЋЈ;А
U
AssignVariableOpAssignVariableOpmodule/embeddings/part_0Identity*
dtype0
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 

save_1/StaticRegexFullMatchStaticRegexFullMatchsave_1/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
c
save_1/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
h
save_1/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
Д
save_1/SelectSelectsave_1/StaticRegexFullMatchsave_1/Const_1save_1/Const_2"/device:CPU:**
T0*
_output_shapes
: 
l
save_1/StringJoin
StringJoinsave_1/Constsave_1/Select"/device:CPU:**
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
Ф
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
Е
save_1/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0"/device:CPU:0*!
_output_shapes
:ЋЈ;А*
dtype0
r
save_1/IdentityIdentitysave_1/Read/ReadVariableOp"/device:CPU:0*
T0*!
_output_shapes
:ЋЈ;А
i
save_1/Identity_1Identitysave_1/Identity"/device:CPU:0*
T0*!
_output_shapes
:ЋЈ;А
|
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB
embeddings
П
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*.
value%B#B973771 128 0,973771:0,128
Ю
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicessave_1/Identity_1"/device:CPU:0*
dtypes
2
®
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
¶
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0
У
save_1/Identity_2Identitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 

save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB
embeddings
Т
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*.
value%B#B973771 128 0,973771:0,128
∞
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*!
_output_shapes
:ЋЈ;А*
dtypes
2
[
save_1/Identity_3Identitysave_1/RestoreV2*
T0*!
_output_shapes
:ЋЈ;А
e
save_1/AssignVariableOpAssignVariableOpmodule/embeddings/part_0save_1/Identity_3*
dtype0
6
save_1/restore_shardNoOp^save_1/AssignVariableOp
1
save_1/restore_allNoOp^save_1/restore_shard
Ж
]dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/ConstConst*
_output_shapes
:*
dtype0*u
valuelBjBar-SABen-USBru-RUBzh-TWBzh-CNBnu-LLBja-JPBko-KRBpt-BRBfr-FRBhe-ILBes-MXBde-DEBhi-IN
Ю
\dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
•
cdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
•
cdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
™
]dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/rangeRangecdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/range/start\dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/Sizecdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/range/delta*
_output_shapes
:
ч
\dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/CastCast]dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
≥
hdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
ъ
mdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_c7b30213-ae29-4991-90c6-3b374462a1dd*
value_dtype0	
№
Бdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2mdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/hash_table]dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/Const\dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/Cast*	
Tin0*

Tout0	
∞
gdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2mdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/hash_tableParseExample/ParseExampleV2:3hdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:€€€€€€€€€
™
_dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
≥
Qdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SparseToDenseSparseToDenseParseExample/ParseExampleV2ParseExample/ParseExampleV2:6gdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/hash_table_Lookup/LookupTableFindV2_dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ц
Qdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ш
Sdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
У
Qdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
в
Kdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hotOneHotQdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SparseToDenseQdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/depthQdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/ConstSdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hot/Const_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
ђ
Ydnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€
®
Gdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SumSumKdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/one_hotYdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Sum/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€
ј
Idnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/ShapeShapeGdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Sum*
T0*
_output_shapes
:
°
Wdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
£
Ydnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
£
Ydnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Б
Qdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/ShapeWdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stackYdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Х
Sdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ѓ
Qdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/strided_sliceSdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
§
Kdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/ReshapeReshapeGdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/SumQdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
ђ
Ydnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
Ж
Sdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/ReshapeReshapeParseExample/ParseExampleV2:9Ydnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape/shape*
T0*#
_output_shapes
:€€€€€€€€€
Ѕ
fdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
«
vdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"Ћџ А   
±
ldnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
А
fdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zerosFillvdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zeros/shape_as_tensorldnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zeros/Const*
T0*!
_output_shapes
:ЋЈ;А
з
Хdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        
ж
Фdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ћџ А   
й
Пdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/SliceSlicefdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/zerosХdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/Slice/beginФdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/Slice/size*
Index0*
T0*!
_output_shapes
:ЋЈ;А
…
rdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0Placeholder*!
_output_shapes
:ЋЈ;А*
dtype0*
shape:ЋЈ;А
№
Уdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpmodule/embeddings/part_0*
_output_shapes
: 
у
ydnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/AssignAssignVariableOpmodule/embeddings/part_0Пdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/PartitionedInitializer/Slice*+
_class!
loc:@module/embeddings/part_0*
dtype0
Р
Жdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embeddings/part_0/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0*+
_class!
loc:@module/embeddings/part_0*!
_output_shapes
:ЋЈ;А*
dtype0
Ї
odnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
с
kdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDims
ExpandDimsSdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshapeodnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
≈
Гdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
Ѓ
}dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/NotEqualNotEqualkdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDimsГdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
≠
|dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/indicesWhere}dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
{dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/valuesGatherNdkdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDims|dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
ђ
Аdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/DenseToSparseTensor/dense_shapeShapekdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ExpandDims*
T0*
_output_shapes
:*
out_type0	
±
odnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/ConstConst*
_output_shapes
: *
dtype0*
value	B B 
И
udnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/StringSplitStringSplitSdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshapeodnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/Const*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:
ї
zdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB B 
д
Иdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsudnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/StringSplitwdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/StringSplit:1wdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/StringSplit:2zdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
≤
idnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/IdentityIdentityИdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/SparseFillEmptyRows/SparseFillEmptyRows*
T0	*'
_output_shapes
:€€€€€€€€€
≤
kdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Identity_1IdentityКdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*#
_output_shapes
:€€€€€€€€€
Х
kdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Identity_2Identitywdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/tokenize/StringSplit:2*
T0	*
_output_shapes
:
Є
vdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
у
ddnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/MaxMaxidnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Identityvdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Max/reduction_indices*
T0	*
_output_shapes
:
®
fdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ї
pdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
≤
pdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_like/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
ы
jdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_likeFillpdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_like/Shapepdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_like/Const*
T0	*
_output_shapes
:
в
ddnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/AddAddddnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Maxjdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/ones_like*
T0	*
_output_shapes
:
ж
hdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/MaximumMaximumfdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Constddnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Add*
T0	*
_output_shapes
:
±
{dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_tablePlaceholder*
_output_shapes
:*
dtype0
Ќ
Бdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
ƒ
Хdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/table_init/asset_filepathConst*
_output_shapes
: *
dtype0*~
valueuBs BmC:\Users\ALEXYA~1\AppData\Local\Temp\tfhub_modules\32f2b2259e1cc8ca58c876921748361283e73997\assets\tokens.txt
Щ
Жdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/table_initInitializeTableFromTextFileV2!module/string_to_index/hash_tableХdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/table_init/asset_filepath*
	key_indexю€€€€€€€€*
value_index€€€€€€€€€
ƒ
Гdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_bucketStringToHashBucketFastkdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Identity_1*#
_output_shapes
:€€€€€€€€€*
num_buckets§
п
Йdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_table_LookupLookupTableFindV2!module/string_to_index/hash_tablekdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/Identity_1Бdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:€€€€€€€€€
Ў
Зdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_table_SizeLookupTableSizeV2!module/string_to_index/hash_table*
_output_shapes
: 
ј
{dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/AddAddГdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_bucketЗdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_table_Size*
T0	*#
_output_shapes
:€€€€€€€€€
Ћ
Аdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/NotEqualNotEqualЙdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_table_LookupБdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/Const*
T0	*#
_output_shapes
:€€€€€€€€€
ї
wdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_LookupSelectАdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/NotEqualЙdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/hash_table_Lookup{dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup/Add*
T0	*#
_output_shapes
:€€€€€€€€€
ё
Мdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
а
Оdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
а
Оdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
®
Жdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_sliceStridedSliceidnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/IdentityМdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stackОdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stack_1Оdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_mask
Ћ
}dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/CastCastЖdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
њ
dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/UniqueUniquewdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index_Lookup*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ъ
Эdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0*!
_output_shapes
:ЋЈ;А*
dtype0
л
Тdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/IdentityIdentityЭdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*
T0*!
_output_shapes
:ЋЈ;А
і
Йdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookupResourceGathermodule/embeddings/part_0dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/Unique*
Tindices0	*≥
_class®
•Ґloc:@dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
dtype0
Ц
Фdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityЙdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup*
T0*≥
_class®
•Ґloc:@dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А
џ
xdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparseSparseSegmentSqrtNФdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/embedding_lookup/Identity_1Бdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/Unique:1}dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse/Cast*
T0*(
_output_shapes
:€€€€€€€€€А
щ
Qdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/ShapeShapexdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse*
T0*
_output_shapes
:
©
_dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ђ
adnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ђ
adnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
©
Ydnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_sliceStridedSliceQdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Shape_dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stackadnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stack_1adnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
†
]dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А
Ћ
[dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1/shapePackYdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/strided_slice]dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1/shape/1*
N*
T0*
_output_shapes
:
к
Udnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1Reshapexdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/embedding_lookup_sparse[dnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1/shape*
T0*(
_output_shapes
:€€€€€€€€€А
ц
Mdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/ConstConst*
_output_shapes
:*
dtype0*u
valuelBj"`	                                              
                           
О
Ldnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
Х
Sdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Sdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
Mdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/rangeRangeSdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/range/startLdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/SizeSdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/range/delta*
_output_shapes
:
„
Ldnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/CastCastMdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
в
Wdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/CastCastMdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/Const*

DstT0	*

SrcT0*
_output_shapes
:
£
Xdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
к
]dnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*@
shared_name1/hash_table_81f1f646-49be-4095-82cf-246777687178*
value_dtype0	
•
qdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2]dnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/hash_tableWdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/CastLdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/Cast*	
Tin0	*

Tout0	
И
_dnn/input_from_feature_columns/input_layer/hour_indicator_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]dnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/hash_tableParseExample/ParseExampleV2:4Xdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/Const*	
Tin0	*

Tout0	*#
_output_shapes
:€€€€€€€€€
Ґ
Wdnn/input_from_feature_columns/input_layer/hour_indicator_1/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
Э
Idnn/input_from_feature_columns/input_layer/hour_indicator_1/SparseToDenseSparseToDenseParseExample/ParseExampleV2:1ParseExample/ParseExampleV2:7_dnn/input_from_feature_columns/input_layer/hour_indicator_1/hash_table_Lookup/LookupTableFindV2Wdnn/input_from_feature_columns/input_layer/hour_indicator_1/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
О
Idnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Р
Kdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
Л
Idnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
Ї
Cdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hotOneHotIdnn/input_from_feature_columns/input_layer/hour_indicator_1/SparseToDenseIdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/depthIdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/ConstKdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hot/Const_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
§
Qdnn/input_from_feature_columns/input_layer/hour_indicator_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€
Р
?dnn/input_from_feature_columns/input_layer/hour_indicator_1/SumSumCdnn/input_from_feature_columns/input_layer/hour_indicator_1/one_hotQdnn/input_from_feature_columns/input_layer/hour_indicator_1/Sum/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€
∞
Adnn/input_from_feature_columns/input_layer/hour_indicator_1/ShapeShape?dnn/input_from_feature_columns/input_layer/hour_indicator_1/Sum*
T0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ы
Qdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ы
Qdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ў
Idnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/hour_indicator_1/ShapeOdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stackQdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Н
Kdnn/input_from_feature_columns/input_layer/hour_indicator_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Ч
Idnn/input_from_feature_columns/input_layer/hour_indicator_1/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/hour_indicator_1/strided_sliceKdnn/input_from_feature_columns/input_layer/hour_indicator_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
М
Cdnn/input_from_feature_columns/input_layer/hour_indicator_1/ReshapeReshape?dnn/input_from_feature_columns/input_layer/hour_indicator_1/SumIdnn/input_from_feature_columns/input_layer/hour_indicator_1/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
»
Odnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/ConstConst*
_output_shapes
:*
dtype0*E
value<B:"0      	                  
            
Р
Ndnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
Ч
Udnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Ч
Udnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
т
Odnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/rangeRangeUdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/range/startNdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/SizeUdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/range/delta*
_output_shapes
:
џ
Ndnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/CastCastOdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
ж
Ydnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/CastCastOdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/Const*

DstT0	*

SrcT0*
_output_shapes
:
•
Zdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
м
_dnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*@
shared_name1/hash_table_f3f64083-cc70-43ef-aceb-6807fe37ea0d*
value_dtype0	
≠
sdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2_dnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/hash_tableYdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/CastNdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/Cast*	
Tin0	*

Tout0	
Н
`dnn/input_from_feature_columns/input_layer/month_indicator_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2_dnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/hash_tableParseExample/ParseExampleV2:5Zdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/Const*	
Tin0	*

Tout0	*#
_output_shapes
:€€€€€€€€€
£
Xdnn/input_from_feature_columns/input_layer/month_indicator_1/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
†
Jdnn/input_from_feature_columns/input_layer/month_indicator_1/SparseToDenseSparseToDenseParseExample/ParseExampleV2:2ParseExample/ParseExampleV2:8`dnn/input_from_feature_columns/input_layer/month_indicator_1/hash_table_Lookup/LookupTableFindV2Xdnn/input_from_feature_columns/input_layer/month_indicator_1/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
П
Jdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
С
Ldnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
М
Jdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
њ
Ddnn/input_from_feature_columns/input_layer/month_indicator_1/one_hotOneHotJdnn/input_from_feature_columns/input_layer/month_indicator_1/SparseToDenseJdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/depthJdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/ConstLdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hot/Const_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
•
Rdnn/input_from_feature_columns/input_layer/month_indicator_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€
У
@dnn/input_from_feature_columns/input_layer/month_indicator_1/SumSumDdnn/input_from_feature_columns/input_layer/month_indicator_1/one_hotRdnn/input_from_feature_columns/input_layer/month_indicator_1/Sum/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€
≤
Bdnn/input_from_feature_columns/input_layer/month_indicator_1/ShapeShape@dnn/input_from_feature_columns/input_layer/month_indicator_1/Sum*
T0*
_output_shapes
:
Ъ
Pdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ь
Rdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ь
Rdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ё
Jdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_sliceStridedSliceBdnn/input_from_feature_columns/input_layer/month_indicator_1/ShapePdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stackRdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stack_1Rdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
О
Ldnn/input_from_feature_columns/input_layer/month_indicator_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Ъ
Jdnn/input_from_feature_columns/input_layer/month_indicator_1/Reshape/shapePackJdnn/input_from_feature_columns/input_layer/month_indicator_1/strided_sliceLdnn/input_from_feature_columns/input_layer/month_indicator_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
П
Ddnn/input_from_feature_columns/input_layer/month_indicator_1/ReshapeReshape@dnn/input_from_feature_columns/input_layer/month_indicator_1/SumJdnn/input_from_feature_columns/input_layer/month_indicator_1/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Б
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
а
1dnn/input_from_feature_columns/input_layer/concatConcatV2Kdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/ReshapeUdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/Reshape_1Cdnn/input_from_feature_columns/input_layer/hour_indicator_1/ReshapeDdnn/input_from_feature_columns/input_layer/month_indicator_1/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*
T0*(
_output_shapes
:€€€€€€€€€≤
Ј
9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
:*
dtype0*
valueB"≤   Р  
©
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *C©–љ
©
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *C©–=
ю
Adnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel* 
_output_shapes
:
≤Р*
dtype0*

seed*
ю
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: 
Т
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel* 
_output_shapes
:
≤Р
Д
3dnn/hiddenlayer_0/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel* 
_output_shapes
:
≤Р
ї
dnn/hiddenlayer_0/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
shape:
≤Р*)
shared_namednn/hiddenlayer_0/kernel
Б
9dnn/hiddenlayer_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel*
_output_shapes
: 
П
dnn/hiddenlayer_0/kernel/AssignAssignVariableOpdnn/hiddenlayer_0/kernel3dnn/hiddenlayer_0/kernel/Initializer/random_uniform*
dtype0
З
,dnn/hiddenlayer_0/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel* 
_output_shapes
:
≤Р*
dtype0
Ґ
(dnn/hiddenlayer_0/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_0/bias*
_output_shapes	
:Р*
dtype0*
valueBР*    
∞
dnn/hiddenlayer_0/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_0/bias*
_output_shapes
: *
dtype0*
shape:Р*'
shared_namednn/hiddenlayer_0/bias
}
7dnn/hiddenlayer_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias*
_output_shapes
: 
А
dnn/hiddenlayer_0/bias/AssignAssignVariableOpdnn/hiddenlayer_0/bias(dnn/hiddenlayer_0/bias/Initializer/zeros*
dtype0
~
*dnn/hiddenlayer_0/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
_output_shapes	
:Р*
dtype0
В
'dnn/hiddenlayer_0/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel* 
_output_shapes
:
≤Р*
dtype0
±
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concat'dnn/hiddenlayer_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р
|
(dnn/hiddenlayer_0/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
_output_shapes	
:Р*
dtype0
Ы
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMul(dnn/hiddenlayer_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€Р
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
T0*
_output_shapes
: *
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R€€€€
А
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
ь
dnn/zero_fraction/condStatelessIfdnn/zero_fraction/LessEqualdnn/hiddenlayer_0/Relu*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *5
else_branch&R$
"dnn_zero_fraction_cond_false_10331*
output_shapes
: : : : : : *4
then_branch%R#
!dnn_zero_fraction_cond_true_10330
d
dnn/zero_fraction/cond/IdentityIdentitydnn/zero_fraction/cond*
T0	*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_1Identitydnn/zero_fraction/cond:1*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_2Identitydnn/zero_fraction/cond:2*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_3Identitydnn/zero_fraction/cond:3*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_4Identitydnn/zero_fraction/cond:4*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_5Identitydnn/zero_fraction/cond:5*
T0*
_output_shapes
: 
Й
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Identity*
T0	*
_output_shapes
: 
Л
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
∞
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ш
.dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_0/fraction_of_zero_values
І
)dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_0/activation
В
dnn/hiddenlayer_0/activationHistogramSummary dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
Ј
9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:*
dtype0*
valueB"Р  n   
©
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *#ёљ
©
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *#ё=
К
Adnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	Рn*
dtype0*

seed**
seed2
ю
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: 
С
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	Рn
Г
3dnn/hiddenlayer_1/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	Рn
Ї
dnn/hiddenlayer_1/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
shape:	Рn*)
shared_namednn/hiddenlayer_1/kernel
Б
9dnn/hiddenlayer_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel*
_output_shapes
: 
П
dnn/hiddenlayer_1/kernel/AssignAssignVariableOpdnn/hiddenlayer_1/kernel3dnn/hiddenlayer_1/kernel/Initializer/random_uniform*
dtype0
Ж
,dnn/hiddenlayer_1/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
_output_shapes
:	Рn*
dtype0
†
(dnn/hiddenlayer_1/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_1/bias*
_output_shapes
:n*
dtype0*
valueBn*    
ѓ
dnn/hiddenlayer_1/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_1/bias*
_output_shapes
: *
dtype0*
shape:n*'
shared_namednn/hiddenlayer_1/bias
}
7dnn/hiddenlayer_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias*
_output_shapes
: 
А
dnn/hiddenlayer_1/bias/AssignAssignVariableOpdnn/hiddenlayer_1/bias(dnn/hiddenlayer_1/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_1/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
_output_shapes
:n*
dtype0
Б
'dnn/hiddenlayer_1/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
_output_shapes
:	Рn*
dtype0
Х
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Relu'dnn/hiddenlayer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€n
{
(dnn/hiddenlayer_1/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
_output_shapes
:n*
dtype0
Ъ
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMul(dnn/hiddenlayer_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€n
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€n
i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R€€€€
Ж
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
Д
dnn/zero_fraction_1/condStatelessIfdnn/zero_fraction_1/LessEqualdnn/hiddenlayer_1/Relu*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *7
else_branch(R&
$dnn_zero_fraction_1_cond_false_10401*
output_shapes
: : : : : : *6
then_branch'R%
#dnn_zero_fraction_1_cond_true_10400
h
!dnn/zero_fraction_1/cond/IdentityIdentitydnn/zero_fraction_1/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_1Identitydnn/zero_fraction_1/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_2Identitydnn/zero_fraction_1/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_3Identitydnn/zero_fraction_1/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_4Identitydnn/zero_fraction_1/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_5Identitydnn/zero_fraction_1/cond:5*
T0*
_output_shapes
: 
П
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Size!dnn/zero_fraction_1/cond/Identity*
T0	*
_output_shapes
: 
П
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

DstT0*

SrcT0	*
_output_shapes
: 
ґ
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ш
.dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_1/fraction_of_zero_values
©
)dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_1/activation
В
dnn/hiddenlayer_1/activationHistogramSummary dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
©
2dnn/logits/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
:*
dtype0*
valueB"n      
Ы
0dnn/logits/kernel/Initializer/random_uniform/minConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
valueB
 *нгgЊ
Ы
0dnn/logits/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
valueB
 *нгg>
ф
:dnn/logits/kernel/Initializer/random_uniform/RandomUniformRandomUniform2dnn/logits/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:n*
dtype0*

seed**
seed2
в
0dnn/logits/kernel/Initializer/random_uniform/subSub0dnn/logits/kernel/Initializer/random_uniform/max0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: 
ф
0dnn/logits/kernel/Initializer/random_uniform/mulMul:dnn/logits/kernel/Initializer/random_uniform/RandomUniform0dnn/logits/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:n
ж
,dnn/logits/kernel/Initializer/random_uniformAdd0dnn/logits/kernel/Initializer/random_uniform/mul0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:n
§
dnn/logits/kernelVarHandleOp*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
shape
:n*"
shared_namednn/logits/kernel
s
2dnn/logits/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel*
_output_shapes
: 
z
dnn/logits/kernel/AssignAssignVariableOpdnn/logits/kernel,dnn/logits/kernel/Initializer/random_uniform*
dtype0
w
%dnn/logits/kernel/Read/ReadVariableOpReadVariableOpdnn/logits/kernel*
_output_shapes

:n*
dtype0
Т
!dnn/logits/bias/Initializer/zerosConst*"
_class
loc:@dnn/logits/bias*
_output_shapes
:*
dtype0*
valueB*    
Ъ
dnn/logits/biasVarHandleOp*"
_class
loc:@dnn/logits/bias*
_output_shapes
: *
dtype0*
shape:* 
shared_namednn/logits/bias
o
0dnn/logits/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias*
_output_shapes
: 
k
dnn/logits/bias/AssignAssignVariableOpdnn/logits/bias!dnn/logits/bias/Initializer/zeros*
dtype0
o
#dnn/logits/bias/Read/ReadVariableOpReadVariableOpdnn/logits/bias*
_output_shapes
:*
dtype0
r
 dnn/logits/MatMul/ReadVariableOpReadVariableOpdnn/logits/kernel*
_output_shapes

:n*
dtype0
З
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Relu dnn/logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
m
!dnn/logits/BiasAdd/ReadVariableOpReadVariableOpdnn/logits/bias*
_output_shapes
:*
dtype0
Е
dnn/logits/BiasAddBiasAdddnn/logits/MatMul!dnn/logits/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_2/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R€€€€
Ж
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
А
dnn/zero_fraction_2/condStatelessIfdnn/zero_fraction_2/LessEqualdnn/logits/BiasAdd*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *7
else_branch(R&
$dnn_zero_fraction_2_cond_false_10470*
output_shapes
: : : : : : *6
then_branch'R%
#dnn_zero_fraction_2_cond_true_10469
h
!dnn/zero_fraction_2/cond/IdentityIdentitydnn/zero_fraction_2/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_1Identitydnn/zero_fraction_2/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_2Identitydnn/zero_fraction_2/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_3Identitydnn/zero_fraction_2/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_4Identitydnn/zero_fraction_2/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_5Identitydnn/zero_fraction_2/cond:5*
T0*
_output_shapes
: 
П
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Size!dnn/zero_fraction_2/cond/Identity*
T0	*
_output_shapes
: 
П
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

DstT0*

SrcT0	*
_output_shapes
: 
ґ
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
К
'dnn/logits/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*3
value*B( B"dnn/logits/fraction_of_zero_values
Ы
"dnn/logits/fraction_of_zero_valuesScalarSummary'dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
o
dnn/logits/activation/tagConst*
_output_shapes
: *
dtype0*&
valueB Bdnn/logits/activation
p
dnn/logits/activationHistogramSummarydnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
S
head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
o
head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
М
head/predictions/class_idsArgMaxdnn/logits/BiasAdd$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:€€€€€€€€€
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Ш
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:€€€€€€€€€
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:€€€€€€€€€
X
head/predictions/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
В
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Х
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
П
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
Ш
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
П
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:€€€€€€€€€
Z
head/predictions/Shape_1Shapednn/logits/BiasAdd*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
М
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Э
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Т
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
Ю
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
У
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:€€€€€€€€€
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
∆
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:€€€€€€€€€

initNoOp
µ
init_all_tablesNoOpВ^dnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/table_init/LookupTableImportV2r^dnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/table_init/LookupTableImportV2t^dnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/table_init/LookupTableImportV2-^module/string_to_index/hash_table/table_init

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
[
save_2/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
dtype0*
shape: 

save_2/StaticRegexFullMatchStaticRegexFullMatchsave_2/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
c
save_2/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
h
save_2/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
Д
save_2/SelectSelectsave_2/StaticRegexFullMatchsave_2/Const_1save_2/Const_2"/device:CPU:**
T0*
_output_shapes
: 
l
save_2/StringJoin
StringJoinsave_2/Constsave_2/Select"/device:CPU:**
N*
_output_shapes
: 
S
save_2/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
m
save_2/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
Ф
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards"/device:CPU:0*
_output_shapes
: 
Е
save_2/Read/ReadVariableOpReadVariableOpmodule/embeddings/part_0"/device:CPU:0*!
_output_shapes
:ЋЈ;А*
dtype0
r
save_2/IdentityIdentitysave_2/Read/ReadVariableOp"/device:CPU:0*
T0*!
_output_shapes
:ЋЈ;А
i
save_2/Identity_1Identitysave_2/Identity"/device:CPU:0*
T0*!
_output_shapes
:ЋЈ;А
Ы
save_2/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*љ
value≥B∞Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_stepBmodule/embeddings
Э
save_2/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*<
value3B1B B B B B B B B973771 128 0,973771:0,128
∆
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slices*dnn/hiddenlayer_0/bias/Read/ReadVariableOp,dnn/hiddenlayer_0/kernel/Read/ReadVariableOp*dnn/hiddenlayer_1/bias/Read/ReadVariableOp,dnn/hiddenlayer_1/kernel/Read/ReadVariableOp#dnn/logits/bias/Read/ReadVariableOp%dnn/logits/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOpsave_2/Identity_1"/device:CPU:0*
dtypes

2	
®
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
¶
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
{
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const"/device:CPU:0
У
save_2/Identity_2Identitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ю
save_2/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*љ
value≥B∞Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_stepBmodule/embeddings
†
!save_2/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*<
value3B1B B B B B B B B973771 128 0,973771:0,128
”
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices"/device:CPU:0*=
_output_shapes+
)::::::::ЋЈ;А*
dtypes

2	
R
save_2/Identity_3Identitysave_2/RestoreV2*
T0*
_output_shapes
:
c
save_2/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/biassave_2/Identity_3*
dtype0
T
save_2/Identity_4Identitysave_2/RestoreV2:1*
T0*
_output_shapes
:
g
save_2/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernelsave_2/Identity_4*
dtype0
T
save_2/Identity_5Identitysave_2/RestoreV2:2*
T0*
_output_shapes
:
e
save_2/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/biassave_2/Identity_5*
dtype0
T
save_2/Identity_6Identitysave_2/RestoreV2:3*
T0*
_output_shapes
:
g
save_2/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernelsave_2/Identity_6*
dtype0
T
save_2/Identity_7Identitysave_2/RestoreV2:4*
T0*
_output_shapes
:
^
save_2/AssignVariableOp_4AssignVariableOpdnn/logits/biassave_2/Identity_7*
dtype0
T
save_2/Identity_8Identitysave_2/RestoreV2:5*
T0*
_output_shapes
:
`
save_2/AssignVariableOp_5AssignVariableOpdnn/logits/kernelsave_2/Identity_8*
dtype0
T
save_2/Identity_9Identitysave_2/RestoreV2:6*
T0	*
_output_shapes
:
Z
save_2/AssignVariableOp_6AssignVariableOpglobal_stepsave_2/Identity_9*
dtype0	
^
save_2/Identity_10Identitysave_2/RestoreV2:7*
T0*!
_output_shapes
:ЋЈ;А
h
save_2/AssignVariableOp_7AssignVariableOpmodule/embeddings/part_0save_2/Identity_10*
dtype0
ъ
save_2/restore_shardNoOp^save_2/AssignVariableOp^save_2/AssignVariableOp_1^save_2/AssignVariableOp_2^save_2/AssignVariableOp_3^save_2/AssignVariableOp_4^save_2/AssignVariableOp_5^save_2/AssignVariableOp_6^save_2/AssignVariableOp_7
1
save_2/restore_allNoOp^save_2/restore_shardЎa
я
о
$dnn_zero_fraction_1_cond_false_104011
-count_nonzero_notequal_dnn_hiddenlayer_1_relu
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosї
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_1_relucount_nonzero/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€n2
count_nonzero/NotEqualН
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:€€€€€€€€€n2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_countЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:€€€€€€€€€n:- )
'
_output_shapes
:€€€€€€€€€n
я
ў
#dnn_zero_fraction_2_cond_true_10469-
)count_nonzero_notequal_dnn_logits_biasadd
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosЈ
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
count_nonzero/NotEqualН
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3Х
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:€€€€€€€€€:- )
'
_output_shapes
:€€€€€€€€€
б
м
"dnn_zero_fraction_cond_false_103311
-count_nonzero_notequal_dnn_hiddenlayer_0_relu
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosЉ
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_0_relucount_nonzero/zeros:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
count_nonzero/NotEqualО
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*(
_output_shapes
:€€€€€€€€€Р2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_countЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*'
_input_shapes
:€€€€€€€€€Р:. *
(
_output_shapes
:€€€€€€€€€Р
„
к
$dnn_zero_fraction_2_cond_false_10470-
)count_nonzero_notequal_dnn_logits_biasadd
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosЈ
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
count_nonzero/NotEqualН
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_countЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:€€€€€€€€€:- )
'
_output_shapes
:€€€€€€€€€
з
Ё
#dnn_zero_fraction_1_cond_true_104001
-count_nonzero_notequal_dnn_hiddenlayer_1_relu
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosї
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_1_relucount_nonzero/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€n2
count_nonzero/NotEqualН
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€n2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3Х
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:€€€€€€€€€n:- )
'
_output_shapes
:€€€€€€€€€n
й
џ
!dnn_zero_fraction_cond_true_103301
-count_nonzero_notequal_dnn_hiddenlayer_0_relu
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zerosЉ
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_0_relucount_nonzero/zeros:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
count_nonzero/NotEqualО
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€Р2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/ConstШ
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastЙ
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValueЛ
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1З
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2Н
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3Х
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*'
_input_shapes
:€€€€€€€€€Р:. *
(
_output_shapes
:€€€€€€€€€Р"D
save_2/Const:0save_2/Identity_2:0save_2/restore_all (5 @F8"р
asset_filepaths№
ў
=module/string_to_index/hash_table/table_init/asset_filepath:0
Чdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/table_init/asset_filepath:0"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"ф
saved_model_assetsЁ*Џ
|
+type.googleapis.com/tensorflow.AssetFileDefM
?
=module/string_to_index/hash_table/table_init/asset_filepath:0
tokens.txt
ў
+type.googleapis.com/tensorflow.AssetFileDef©
Ъ
Чdnn/input_from_feature_columns/input_layer/TitleText_hub_module_embedding_1/module_apply_default/string_to_index/hash_table/table_init/asset_filepath:0
tokens.txt"%
saved_model_main_op


group_deps"к
	summaries№
ў
+dnn/hiddenlayer_0/fraction_of_zero_values:0
dnn/hiddenlayer_0/activation:0
+dnn/hiddenlayer_1/fraction_of_zero_values:0
dnn/hiddenlayer_1/activation:0
$dnn/logits/fraction_of_zero_values:0
dnn/logits/activation:0"≥
table_initializerЭ
Ъ
,module/string_to_index/hash_table/table_init
Бdnn/input_from_feature_columns/input_layer/LanguageCode_indicator_1/LanguageCode_lookup/hash_table/table_init/LookupTableImportV2
qdnn/input_from_feature_columns/input_layer/hour_indicator_1/hour_lookup/hash_table/table_init/LookupTableImportV2
sdnn/input_from_feature_columns/input_layer/month_indicator_1/month_lookup/hash_table/table_init/LookupTableImportV2"±
trainable_variablesЩЦ
®
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
Ч
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
®
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
Ч
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
М
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08"Є	
	variables™	І	
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
°
module/embeddings/part_0:0AssignVariableOp.module/embeddings/part_0/Read/ReadVariableOp:0"%
module/embeddingsЋЈ;А  "ЋЈ;А(2checkpoint_initializer:0
®
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
Ч
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
®
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
Ч
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
М
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08*„
classificationƒ
3
inputs)
input_example_tensor:0€€€€€€€€€-
classes"
head/Tile:0€€€€€€€€€A
scores7
 head/predictions/probabilities:0€€€€€€€€€tensorflow/serving/classify*з
predictџ
5
examples)
input_example_tensor:0€€€€€€€€€?
all_class_ids.
head/predictions/Tile:0€€€€€€€€€?
all_classes0
head/predictions/Tile_1:0€€€€€€€€€A
	class_ids4
head/predictions/ExpandDims:0	€€€€€€€€€@
classes5
head/predictions/str_classes:0€€€€€€€€€5
logits+
dnn/logits/BiasAdd:0€€€€€€€€€H
probabilities7
 head/predictions/probabilities:0€€€€€€€€€tensorflow/serving/predict*Ў
serving_defaultƒ
3
inputs)
input_example_tensor:0€€€€€€€€€-
classes"
head/Tile:0€€€€€€€€€A
scores7
 head/predictions/probabilities:0€€€€€€€€€tensorflow/serving/classify

Z
default_data_placeholderPlaceholder*$
shape:?????????*
dtype0
r
conv2d_2_conv2d_kernel
VariableV2*
shape: *
shared_name *
dtype0*
	container 
d
conv2d_2_conv2d_bias
VariableV2*
shape: *
shared_name *
dtype0*
	container 
B
ConstConst*%
valueB"             *
dtype0
D
Const_1Const*%
valueB	"               *
dtype0	
g
StatelessTruncatedNormalStatelessTruncatedNormalConstConst_1*
T0*
Tseed0	*
dtype0
8
Const_2Const*
valueB 2?m?7&??*
dtype0
=
CastCastConst_2*

SrcT0*
Truncate( *

DstT0
K
Init_conv2d_2_conv2d_kernelMulStatelessTruncatedNormalCast*
T0
?
Assign_conv2d_2_conv2d_kernelAssignconv2d_2_conv2d_kernelInit_conv2d_2_conv2d_kernel*
use_locking(*
T0*
validate_shape(
5
Const_3Const*
valueB: *
dtype0
D
Const_4Const*%
valueB	"               *
dtype0	
k
StatelessTruncatedNormal_1StatelessTruncatedNormalConst_3Const_4*
T0*
Tseed0	*
dtype0
8
Const_5Const*
valueB 2?m?7&??*
dtype0
?
Cast_1CastConst_5*

SrcT0*
Truncate( *

DstT0
M
Init_conv2d_2_conv2d_biasMulStatelessTruncatedNormal_1Cast_1*
T0
?
Assign_conv2d_2_conv2d_biasAssignconv2d_2_conv2d_biasInit_conv2d_2_conv2d_bias*
use_locking(*
T0*
validate_shape(
r
conv2d_4_conv2d_kernel
VariableV2*
shape: @*
shared_name *
dtype0*
	container 
d
conv2d_4_conv2d_bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
D
Const_6Const*%
valueB"          @   *
dtype0
D
Const_7Const*%
valueB	"               *
dtype0	
k
StatelessTruncatedNormal_2StatelessTruncatedNormalConst_6Const_7*
T0*
Tseed0	*
dtype0
8
Const_8Const*
valueB 2?V??@??*
dtype0
?
Cast_2CastConst_8*

SrcT0*
Truncate( *

DstT0
O
Init_conv2d_4_conv2d_kernelMulStatelessTruncatedNormal_2Cast_2*
T0
?
Assign_conv2d_4_conv2d_kernelAssignconv2d_4_conv2d_kernelInit_conv2d_4_conv2d_kernel*
use_locking(*
T0*
validate_shape(
5
Const_9Const*
valueB:@*
dtype0
E
Const_10Const*%
valueB	"               *
dtype0	
l
StatelessTruncatedNormal_3StatelessTruncatedNormalConst_9Const_10*
T0*
Tseed0	*
dtype0
9
Const_11Const*
valueB 2?V??@??*
dtype0
@
Cast_3CastConst_11*

SrcT0*
Truncate( *

DstT0
M
Init_conv2d_4_conv2d_biasMulStatelessTruncatedNormal_3Cast_3*
T0
?
Assign_conv2d_4_conv2d_biasAssignconv2d_4_conv2d_biasInit_conv2d_4_conv2d_bias*
use_locking(*
T0*
validate_shape(
s
conv2d_6_conv2d_kernel
VariableV2*
shape:@?*
shared_name *
dtype0*
	container 
e
conv2d_6_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_12Const*%
valueB"      @   ?   *
dtype0
E
Const_13Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_4StatelessTruncatedNormalConst_12Const_13*
T0*
Tseed0	*
dtype0
9
Const_14Const*
valueB 2?m?7&??*
dtype0
@
Cast_4CastConst_14*

SrcT0*
Truncate( *

DstT0
O
Init_conv2d_6_conv2d_kernelMulStatelessTruncatedNormal_4Cast_4*
T0
?
Assign_conv2d_6_conv2d_kernelAssignconv2d_6_conv2d_kernelInit_conv2d_6_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_15Const*
valueB:?*
dtype0
E
Const_16Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_5StatelessTruncatedNormalConst_15Const_16*
T0*
Tseed0	*
dtype0
9
Const_17Const*
valueB 2?m?7&??*
dtype0
@
Cast_5CastConst_17*

SrcT0*
Truncate( *

DstT0
M
Init_conv2d_6_conv2d_biasMulStatelessTruncatedNormal_5Cast_5*
T0
?
Assign_conv2d_6_conv2d_biasAssignconv2d_6_conv2d_biasInit_conv2d_6_conv2d_bias*
use_locking(*
T0*
validate_shape(
t
conv2d_7_conv2d_kernel
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
e
conv2d_7_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_18Const*%
valueB"      ?   ?   *
dtype0
E
Const_19Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_6StatelessTruncatedNormalConst_18Const_19*
T0*
Tseed0	*
dtype0
9
Const_20Const*
valueB 2?V??@??*
dtype0
@
Cast_6CastConst_20*

SrcT0*
Truncate( *

DstT0
O
Init_conv2d_7_conv2d_kernelMulStatelessTruncatedNormal_6Cast_6*
T0
?
Assign_conv2d_7_conv2d_kernelAssignconv2d_7_conv2d_kernelInit_conv2d_7_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_21Const*
valueB:?*
dtype0
E
Const_22Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_7StatelessTruncatedNormalConst_21Const_22*
T0*
Tseed0	*
dtype0
9
Const_23Const*
valueB 2?V??@??*
dtype0
@
Cast_7CastConst_23*

SrcT0*
Truncate( *

DstT0
M
Init_conv2d_7_conv2d_biasMulStatelessTruncatedNormal_7Cast_7*
T0
?
Assign_conv2d_7_conv2d_biasAssignconv2d_7_conv2d_biasInit_conv2d_7_conv2d_bias*
use_locking(*
T0*
validate_shape(
t
conv2d_9_conv2d_kernel
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
e
conv2d_9_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_24Const*%
valueB"      ?      *
dtype0
E
Const_25Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_8StatelessTruncatedNormalConst_24Const_25*
T0*
Tseed0	*
dtype0
9
Const_26Const*
valueB 2?V??@??*
dtype0
@
Cast_8CastConst_26*

SrcT0*
Truncate( *

DstT0
O
Init_conv2d_9_conv2d_kernelMulStatelessTruncatedNormal_8Cast_8*
T0
?
Assign_conv2d_9_conv2d_kernelAssignconv2d_9_conv2d_kernelInit_conv2d_9_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_27Const*
valueB:?*
dtype0
E
Const_28Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_9StatelessTruncatedNormalConst_27Const_28*
T0*
Tseed0	*
dtype0
9
Const_29Const*
valueB 2?V??@??*
dtype0
@
Cast_9CastConst_29*

SrcT0*
Truncate( *

DstT0
M
Init_conv2d_9_conv2d_biasMulStatelessTruncatedNormal_9Cast_9*
T0
?
Assign_conv2d_9_conv2d_biasAssignconv2d_9_conv2d_biasInit_conv2d_9_conv2d_bias*
use_locking(*
T0*
validate_shape(
u
conv2d_10_conv2d_kernel
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
f
conv2d_10_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_30Const*%
valueB"            *
dtype0
E
Const_31Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_10StatelessTruncatedNormalConst_30Const_31*
T0*
Tseed0	*
dtype0
9
Const_32Const*
valueB 2?m?7&??*
dtype0
A
Cast_10CastConst_32*

SrcT0*
Truncate( *

DstT0
R
Init_conv2d_10_conv2d_kernelMulStatelessTruncatedNormal_10Cast_10*
T0
?
Assign_conv2d_10_conv2d_kernelAssignconv2d_10_conv2d_kernelInit_conv2d_10_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_33Const*
valueB:?*
dtype0
E
Const_34Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_11StatelessTruncatedNormalConst_33Const_34*
T0*
Tseed0	*
dtype0
9
Const_35Const*
valueB 2?m?7&??*
dtype0
A
Cast_11CastConst_35*

SrcT0*
Truncate( *

DstT0
P
Init_conv2d_10_conv2d_biasMulStatelessTruncatedNormal_11Cast_11*
T0
?
Assign_conv2d_10_conv2d_biasAssignconv2d_10_conv2d_biasInit_conv2d_10_conv2d_bias*
use_locking(*
T0*
validate_shape(
u
conv2d_12_conv2d_kernel
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
f
conv2d_12_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_36Const*%
valueB"         ?   *
dtype0
E
Const_37Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_12StatelessTruncatedNormalConst_36Const_37*
T0*
Tseed0	*
dtype0
9
Const_38Const*
valueB 2?m?7&??*
dtype0
A
Cast_12CastConst_38*

SrcT0*
Truncate( *

DstT0
R
Init_conv2d_12_conv2d_kernelMulStatelessTruncatedNormal_12Cast_12*
T0
?
Assign_conv2d_12_conv2d_kernelAssignconv2d_12_conv2d_kernelInit_conv2d_12_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_39Const*
valueB:?*
dtype0
E
Const_40Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_13StatelessTruncatedNormalConst_39Const_40*
T0*
Tseed0	*
dtype0
9
Const_41Const*
valueB 2?m?7&??*
dtype0
A
Cast_13CastConst_41*

SrcT0*
Truncate( *

DstT0
P
Init_conv2d_12_conv2d_biasMulStatelessTruncatedNormal_13Cast_13*
T0
?
Assign_conv2d_12_conv2d_biasAssignconv2d_12_conv2d_biasInit_conv2d_12_conv2d_bias*
use_locking(*
T0*
validate_shape(
u
conv2d_13_conv2d_kernel
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
f
conv2d_13_conv2d_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
E
Const_42Const*%
valueB"      ?   ?   *
dtype0
E
Const_43Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_14StatelessTruncatedNormalConst_42Const_43*
T0*
Tseed0	*
dtype0
9
Const_44Const*
valueB 2?V??@??*
dtype0
A
Cast_14CastConst_44*

SrcT0*
Truncate( *

DstT0
R
Init_conv2d_13_conv2d_kernelMulStatelessTruncatedNormal_14Cast_14*
T0
?
Assign_conv2d_13_conv2d_kernelAssignconv2d_13_conv2d_kernelInit_conv2d_13_conv2d_kernel*
use_locking(*
T0*
validate_shape(
7
Const_45Const*
valueB:?*
dtype0
E
Const_46Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_15StatelessTruncatedNormalConst_45Const_46*
T0*
Tseed0	*
dtype0
9
Const_47Const*
valueB 2?V??@??*
dtype0
A
Cast_15CastConst_47*

SrcT0*
Truncate( *

DstT0
P
Init_conv2d_13_conv2d_biasMulStatelessTruncatedNormal_15Cast_15*
T0
?
Assign_conv2d_13_conv2d_biasAssignconv2d_13_conv2d_biasInit_conv2d_13_conv2d_bias*
use_locking(*
T0*
validate_shape(
=
Const_48Const*
valueB"?????   *
dtype0
k
dense_16_dense_kernel
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
d
dense_16_dense_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
=
Const_49Const*
valueB"?      *
dtype0
E
Const_50Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_16StatelessTruncatedNormalConst_49Const_50*
T0*
Tseed0	*
dtype0
9
Const_51Const*
valueB 2?????0??*
dtype0
A
Cast_16CastConst_51*

SrcT0*
Truncate( *

DstT0
P
Init_dense_16_dense_kernelMulStatelessTruncatedNormal_16Cast_16*
T0
?
Assign_dense_16_dense_kernelAssigndense_16_dense_kernelInit_dense_16_dense_kernel*
use_locking(*
T0*
validate_shape(
7
Const_52Const*
valueB:?*
dtype0
E
Const_53Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_17StatelessTruncatedNormalConst_52Const_53*
T0*
Tseed0	*
dtype0
9
Const_54Const*
valueB 2?????0??*
dtype0
A
Cast_17CastConst_54*

SrcT0*
Truncate( *

DstT0
N
Init_dense_16_dense_biasMulStatelessTruncatedNormal_17Cast_17*
T0
?
Assign_dense_16_dense_biasAssigndense_16_dense_biasInit_dense_16_dense_bias*
use_locking(*
T0*
validate_shape(
k
dense_17_dense_kernel
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
d
dense_17_dense_bias
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
=
Const_55Const*
valueB"   ?  *
dtype0
E
Const_56Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_18StatelessTruncatedNormalConst_55Const_56*
T0*
Tseed0	*
dtype0
9
Const_57Const*
valueB 2?????0??*
dtype0
A
Cast_18CastConst_57*

SrcT0*
Truncate( *

DstT0
P
Init_dense_17_dense_kernelMulStatelessTruncatedNormal_18Cast_18*
T0
?
Assign_dense_17_dense_kernelAssigndense_17_dense_kernelInit_dense_17_dense_kernel*
use_locking(*
T0*
validate_shape(
7
Const_58Const*
valueB:?*
dtype0
E
Const_59Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_19StatelessTruncatedNormalConst_58Const_59*
T0*
Tseed0	*
dtype0
9
Const_60Const*
valueB 2?????0??*
dtype0
A
Cast_19CastConst_60*

SrcT0*
Truncate( *

DstT0
N
Init_dense_17_dense_biasMulStatelessTruncatedNormal_19Cast_19*
T0
?
Assign_dense_17_dense_biasAssigndense_17_dense_biasInit_dense_17_dense_bias*
use_locking(*
T0*
validate_shape(
j
dense_18_dense_kernel
VariableV2*
shape:	?
*
shared_name *
dtype0*
	container 
c
dense_18_dense_bias
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
=
Const_61Const*
valueB"?  
   *
dtype0
E
Const_62Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_20StatelessTruncatedNormalConst_61Const_62*
T0*
Tseed0	*
dtype0
9
Const_63Const*
valueB 2n?????*
dtype0
A
Cast_20CastConst_63*

SrcT0*
Truncate( *

DstT0
P
Init_dense_18_dense_kernelMulStatelessTruncatedNormal_20Cast_20*
T0
?
Assign_dense_18_dense_kernelAssigndense_18_dense_kernelInit_dense_18_dense_kernel*
use_locking(*
T0*
validate_shape(
6
Const_64Const*
valueB:
*
dtype0
E
Const_65Const*%
valueB	"               *
dtype0	
n
StatelessTruncatedNormal_21StatelessTruncatedNormalConst_64Const_65*
T0*
Tseed0	*
dtype0
9
Const_66Const*
valueB 2n?????*
dtype0
A
Cast_21CastConst_66*

SrcT0*
Truncate( *

DstT0
N
Init_dense_18_dense_biasMulStatelessTruncatedNormal_21Cast_21*
T0
?
Assign_dense_18_dense_biasAssigndense_18_dense_biasInit_dense_18_dense_bias*
use_locking(*
T0*
validate_shape(
6
PlaceholderPlaceholder*
shape:*
dtype0
7
numberOfLossesPlaceholder*
shape: *
dtype0
1
trainingPlaceholder*
shape: *
dtype0

?
Conv2dConv2Ddefault_data_placeholderconv2d_2_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
P
BiasAddBiasAddConv2dconv2d_2_conv2d_bias*
T0*
data_formatNHWC

ReluReluBiasAdd*
T0
.
Activation_conv2d_2IdentityRelu*
T0
E
Const_67Const*%
valueB"            *
dtype0
E
Const_68Const*%
valueB"            *
dtype0
p
MaxPool	MaxPoolV2Activation_conv2d_2Const_67Const_68*
paddingSAME*
T0*
data_formatNHWC
?
Conv2d_1Conv2DMaxPoolconv2d_4_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
T
	BiasAdd_1BiasAddConv2d_1conv2d_4_conv2d_bias*
T0*
data_formatNHWC
"
Relu_1Relu	BiasAdd_1*
T0
0
Activation_conv2d_4IdentityRelu_1*
T0
E
Const_69Const*%
valueB"            *
dtype0
E
Const_70Const*%
valueB"            *
dtype0
r
	MaxPool_1	MaxPoolV2Activation_conv2d_4Const_69Const_70*
paddingSAME*
T0*
data_formatNHWC
?
Conv2d_2Conv2D	MaxPool_1conv2d_6_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
T
	BiasAdd_2BiasAddConv2d_2conv2d_6_conv2d_bias*
T0*
data_formatNHWC
"
Relu_2Relu	BiasAdd_2*
T0
0
Activation_conv2d_6IdentityRelu_2*
T0
?
Conv2d_3Conv2DActivation_conv2d_6conv2d_7_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
T
	BiasAdd_3BiasAddConv2d_3conv2d_7_conv2d_bias*
T0*
data_formatNHWC
"
Relu_3Relu	BiasAdd_3*
T0
0
Activation_conv2d_7IdentityRelu_3*
T0
E
Const_71Const*%
valueB"            *
dtype0
E
Const_72Const*%
valueB"            *
dtype0
r
	MaxPool_2	MaxPoolV2Activation_conv2d_7Const_71Const_72*
paddingSAME*
T0*
data_formatNHWC
?
Conv2d_4Conv2D	MaxPool_2conv2d_9_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
T
	BiasAdd_4BiasAddConv2d_4conv2d_9_conv2d_bias*
T0*
data_formatNHWC
"
Relu_4Relu	BiasAdd_4*
T0
0
Activation_conv2d_9IdentityRelu_4*
T0
?
Conv2d_5Conv2DActivation_conv2d_9conv2d_10_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
U
	BiasAdd_5BiasAddConv2d_5conv2d_10_conv2d_bias*
T0*
data_formatNHWC
"
Relu_5Relu	BiasAdd_5*
T0
1
Activation_conv2d_10IdentityRelu_5*
T0
E
Const_73Const*%
valueB"            *
dtype0
E
Const_74Const*%
valueB"            *
dtype0
s
	MaxPool_3	MaxPoolV2Activation_conv2d_10Const_73Const_74*
paddingSAME*
T0*
data_formatNHWC
?
Conv2d_6Conv2D	MaxPool_3conv2d_12_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
U
	BiasAdd_6BiasAddConv2d_6conv2d_12_conv2d_bias*
T0*
data_formatNHWC
"
Relu_6Relu	BiasAdd_6*
T0
1
Activation_conv2d_12IdentityRelu_6*
T0
?
Conv2d_7Conv2DActivation_conv2d_12conv2d_13_conv2d_kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
U
	BiasAdd_7BiasAddConv2d_7conv2d_13_conv2d_bias*
T0*
data_formatNHWC
"
Relu_7Relu	BiasAdd_7*
T0
1
Activation_conv2d_13IdentityRelu_7*
T0
E
Const_75Const*%
valueB"            *
dtype0
E
Const_76Const*%
valueB"            *
dtype0
s
	MaxPool_4	MaxPoolV2Activation_conv2d_13Const_75Const_76*
paddingSAME*
T0*
data_formatNHWC
>
ReshapeReshape	MaxPool_4Const_48*
T0*
Tshape0
_
MatMulMatMulReshapedense_16_dense_kernel*
transpose_b( *
T0*
transpose_a( 
0
AddAddMatMuldense_16_dense_bias*
T0

Relu_8ReluAdd*
T0
0
Activation_dense_16IdentityRelu_8*
T0
m
MatMul_1MatMulActivation_dense_16dense_17_dense_kernel*
transpose_b( *
T0*
transpose_a( 
4
Add_1AddMatMul_1dense_17_dense_bias*
T0

Relu_9ReluAdd_1*
T0
0
Activation_dense_17IdentityRelu_9*
T0
m
MatMul_2MatMulActivation_dense_17dense_18_dense_kernel*
transpose_b( *
T0*
transpose_a( 
4
Add_2AddMatMul_2dense_18_dense_bias*
T0
/
Activation_dense_18IdentityAdd_2*
T0
i
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsActivation_dense_18Placeholder*
T0
2
Const_77Const*
value	B : *
dtype0
l
default_training_lossMeanSoftmaxCrossEntropyWithLogitsConst_77*

Tidx0*
	keep_dims( *
T0
>
Gradients/OnesLikeOnesLikedefault_training_loss*
T0
P
Gradients/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0
9
Gradients/ConstConst*
value	B : *
dtype0
;
Gradients/Const_1Const*
value	B :*
dtype0
@
Gradients/SizeSizeGradients/Shape*
T0*
out_type0
7
Gradients/AddAddConst_77Gradients/Size*
T0
<
Gradients/ModModGradients/AddGradients/Size*
T0
X
Gradients/RangeRangeGradients/ConstGradients/SizeGradients/Const_1*

Tidx0
8
Gradients/OnesLike_1OnesLikeGradients/Mod*
T0
?
Gradients/DynamicStitchDynamicStitchGradients/RangeGradients/ModGradients/ShapeGradients/OnesLike_1*
T0*
N
;
Gradients/Const_2Const*
value	B :*
dtype0
Q
Gradients/MaximumMaximumGradients/DynamicStitchGradients/Const_2*
T0
A
Gradients/DivDivGradients/ShapeGradients/Maximum*
T0
`
Gradients/ReshapeReshapeGradients/OnesLikeGradients/DynamicStitch*
T0*
Tshape0
S
Gradients/TileTileGradients/ReshapeGradients/Div*

Tmultiples0*
T0
R
Gradients/Shape_1ShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0
J
Gradients/Shape_2Shapedefault_training_loss*
T0*
out_type0
;
Gradients/Const_3Const*
value	B : *
dtype0
b
Gradients/ProdProdGradients/Shape_2Gradients/Const_3*

Tidx0*
	keep_dims( *
T0
d
Gradients/Prod_1ProdGradients/Shape_1Gradients/Const_3*

Tidx0*
	keep_dims( *
T0
;
Gradients/Const_4Const*
value	B :*
dtype0
J
Gradients/Maximum_1MaximumGradients/ProdGradients/Const_4*
T0
F
Gradients/Div_1DivGradients/Prod_1Gradients/Maximum_1*
T0
O
Gradients/CastCastGradients/Div_1*

SrcT0*
Truncate( *

DstT0
?
Gradients/Div_2DivGradients/TileGradients/Cast*
T0
J
Gradients/ZerosLike	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0
J
Gradients/Const_5/ConstConst*
valueB :
?????????*
dtype0
a
Gradients/ExpandDims
ExpandDimsGradients/Div_2Gradients/Const_5/Const*

Tdim0*
T0
Y
Gradients/MultiplyMulGradients/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0
@
Gradients/LogSoftmax
LogSoftmaxActivation_dense_18*
T0
D
Gradients/Const_6/ConstConst*
valueB
 *  ??*
dtype0
S
Gradients/Multiply_1MulGradients/LogSoftmaxGradients/Const_6/Const*
T0
J
Gradients/Const_7/ConstConst*
valueB :
?????????*
dtype0
c
Gradients/ExpandDims_1
ExpandDimsGradients/Div_2Gradients/Const_7/Const*

Tdim0*
T0
R
Gradients/Multiply_2MulGradients/ExpandDims_1Gradients/Multiply_1*
T0
;
Gradients/IdentityIdentityGradients/Multiply*
T0
=
Gradients/Identity_1IdentityGradients/Identity*
T0
=
Gradients/Identity_2IdentityGradients/Identity*
T0
=
Gradients/Shape_3ShapeMatMul_2*
T0*
out_type0
H
Gradients/Shape_4Shapedense_18_dense_bias*
T0*
out_type0
g
Gradients/BroadcastGradientArgsBroadcastGradientArgsGradients/Shape_3Gradients/Shape_4*
T0
q
Gradients/SumSumGradients/Identity_1Gradients/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
W
Gradients/Reshape_1ReshapeGradients/SumGradients/Shape_3*
T0*
Tshape0
u
Gradients/Sum_1SumGradients/Identity_2!Gradients/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_2ReshapeGradients/Sum_1Gradients/Shape_4*
T0*
Tshape0
u
Gradients/MatMulMatMulGradients/Reshape_1dense_18_dense_kernel*
transpose_b(*
T0*
transpose_a( 
u
Gradients/MatMul_1MatMulActivation_dense_17Gradients/Reshape_1*
transpose_b( *
T0*
transpose_a(
;
Gradients/Identity_3IdentityGradients/MatMul*
T0
D
Gradients/ReluGradReluGradGradients/Identity_3Add_1*
T0
=
Gradients/Identity_4IdentityGradients/ReluGrad*
T0
=
Gradients/Identity_5IdentityGradients/ReluGrad*
T0
=
Gradients/Shape_5ShapeMatMul_1*
T0*
out_type0
H
Gradients/Shape_6Shapedense_17_dense_bias*
T0*
out_type0
i
!Gradients/BroadcastGradientArgs_1BroadcastGradientArgsGradients/Shape_5Gradients/Shape_6*
T0
u
Gradients/Sum_2SumGradients/Identity_4!Gradients/BroadcastGradientArgs_1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_3ReshapeGradients/Sum_2Gradients/Shape_5*
T0*
Tshape0
w
Gradients/Sum_3SumGradients/Identity_5#Gradients/BroadcastGradientArgs_1:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_4ReshapeGradients/Sum_3Gradients/Shape_6*
T0*
Tshape0
w
Gradients/MatMul_2MatMulGradients/Reshape_3dense_17_dense_kernel*
transpose_b(*
T0*
transpose_a( 
u
Gradients/MatMul_3MatMulActivation_dense_16Gradients/Reshape_3*
transpose_b( *
T0*
transpose_a(
=
Gradients/Identity_6IdentityGradients/MatMul_2*
T0
D
Gradients/ReluGrad_1ReluGradGradients/Identity_6Add*
T0
?
Gradients/Identity_7IdentityGradients/ReluGrad_1*
T0
?
Gradients/Identity_8IdentityGradients/ReluGrad_1*
T0
;
Gradients/Shape_7ShapeMatMul*
T0*
out_type0
H
Gradients/Shape_8Shapedense_16_dense_bias*
T0*
out_type0
i
!Gradients/BroadcastGradientArgs_2BroadcastGradientArgsGradients/Shape_7Gradients/Shape_8*
T0
u
Gradients/Sum_4SumGradients/Identity_7!Gradients/BroadcastGradientArgs_2*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_5ReshapeGradients/Sum_4Gradients/Shape_7*
T0*
Tshape0
w
Gradients/Sum_5SumGradients/Identity_8#Gradients/BroadcastGradientArgs_2:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_6ReshapeGradients/Sum_5Gradients/Shape_8*
T0*
Tshape0
w
Gradients/MatMul_4MatMulGradients/Reshape_5dense_16_dense_kernel*
transpose_b(*
T0*
transpose_a( 
i
Gradients/MatMul_5MatMulReshapeGradients/Reshape_5*
transpose_b( *
T0*
transpose_a(
>
Gradients/Shape_9Shape	MaxPool_4*
T0*
out_type0
\
Gradients/Reshape_7ReshapeGradients/MatMul_4Gradients/Shape_9*
T0*
Tshape0
?
Gradients/MaxPoolGradV2MaxPoolGradV2Activation_conv2d_13	MaxPool_4Gradients/Reshape_7Const_75Const_76*
paddingSAME*
T0*
data_formatNHWC
B
Gradients/Identity_9IdentityGradients/MaxPoolGradV2*
T0
J
Gradients/ReluGrad_2ReluGradGradients/Identity_9	BiasAdd_7*
T0
Z
Gradients/BiasAddGradBiasAddGradGradients/ReluGrad_2*
T0*
data_formatNHWC
@
Gradients/Identity_10IdentityGradients/ReluGrad_2*
T0
J
Gradients/Shape_10ShapeActivation_conv2d_12*
T0*
out_type0
?
Gradients/Conv2DBackpropInputConv2DBackpropInputGradients/Shape_10conv2d_13_conv2d_kernelGradients/Identity_10*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
M
Gradients/Shape_11Shapeconv2d_13_conv2d_kernel*
T0*
out_type0
?
Gradients/Conv2DBackpropFilterConv2DBackpropFilterActivation_conv2d_12Gradients/Shape_11Gradients/Identity_10*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
I
Gradients/Identity_11IdentityGradients/Conv2DBackpropInput*
T0
K
Gradients/ReluGrad_3ReluGradGradients/Identity_11	BiasAdd_6*
T0
\
Gradients/BiasAddGrad_1BiasAddGradGradients/ReluGrad_3*
T0*
data_formatNHWC
@
Gradients/Identity_12IdentityGradients/ReluGrad_3*
T0
?
Gradients/Shape_12Shape	MaxPool_3*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_1Conv2DBackpropInputGradients/Shape_12conv2d_12_conv2d_kernelGradients/Identity_12*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
M
Gradients/Shape_13Shapeconv2d_12_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_1Conv2DBackpropFilter	MaxPool_3Gradients/Shape_13Gradients/Identity_12*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
?
Gradients/MaxPoolGradV2_1MaxPoolGradV2Activation_conv2d_10	MaxPool_3Gradients/Conv2DBackpropInput_1Const_73Const_74*
paddingSAME*
T0*
data_formatNHWC
E
Gradients/Identity_13IdentityGradients/MaxPoolGradV2_1*
T0
K
Gradients/ReluGrad_4ReluGradGradients/Identity_13	BiasAdd_5*
T0
\
Gradients/BiasAddGrad_2BiasAddGradGradients/ReluGrad_4*
T0*
data_formatNHWC
@
Gradients/Identity_14IdentityGradients/ReluGrad_4*
T0
I
Gradients/Shape_14ShapeActivation_conv2d_9*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_2Conv2DBackpropInputGradients/Shape_14conv2d_10_conv2d_kernelGradients/Identity_14*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
M
Gradients/Shape_15Shapeconv2d_10_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_2Conv2DBackpropFilterActivation_conv2d_9Gradients/Shape_15Gradients/Identity_14*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
K
Gradients/Identity_15IdentityGradients/Conv2DBackpropInput_2*
T0
K
Gradients/ReluGrad_5ReluGradGradients/Identity_15	BiasAdd_4*
T0
\
Gradients/BiasAddGrad_3BiasAddGradGradients/ReluGrad_5*
T0*
data_formatNHWC
@
Gradients/Identity_16IdentityGradients/ReluGrad_5*
T0
?
Gradients/Shape_16Shape	MaxPool_2*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_3Conv2DBackpropInputGradients/Shape_16conv2d_9_conv2d_kernelGradients/Identity_16*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
L
Gradients/Shape_17Shapeconv2d_9_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_3Conv2DBackpropFilter	MaxPool_2Gradients/Shape_17Gradients/Identity_16*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
?
Gradients/MaxPoolGradV2_2MaxPoolGradV2Activation_conv2d_7	MaxPool_2Gradients/Conv2DBackpropInput_3Const_71Const_72*
paddingSAME*
T0*
data_formatNHWC
E
Gradients/Identity_17IdentityGradients/MaxPoolGradV2_2*
T0
K
Gradients/ReluGrad_6ReluGradGradients/Identity_17	BiasAdd_3*
T0
\
Gradients/BiasAddGrad_4BiasAddGradGradients/ReluGrad_6*
T0*
data_formatNHWC
@
Gradients/Identity_18IdentityGradients/ReluGrad_6*
T0
I
Gradients/Shape_18ShapeActivation_conv2d_6*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_4Conv2DBackpropInputGradients/Shape_18conv2d_7_conv2d_kernelGradients/Identity_18*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
L
Gradients/Shape_19Shapeconv2d_7_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_4Conv2DBackpropFilterActivation_conv2d_6Gradients/Shape_19Gradients/Identity_18*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
K
Gradients/Identity_19IdentityGradients/Conv2DBackpropInput_4*
T0
K
Gradients/ReluGrad_7ReluGradGradients/Identity_19	BiasAdd_2*
T0
\
Gradients/BiasAddGrad_5BiasAddGradGradients/ReluGrad_7*
T0*
data_formatNHWC
@
Gradients/Identity_20IdentityGradients/ReluGrad_7*
T0
?
Gradients/Shape_20Shape	MaxPool_1*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_5Conv2DBackpropInputGradients/Shape_20conv2d_6_conv2d_kernelGradients/Identity_20*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
L
Gradients/Shape_21Shapeconv2d_6_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_5Conv2DBackpropFilter	MaxPool_1Gradients/Shape_21Gradients/Identity_20*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
?
Gradients/MaxPoolGradV2_3MaxPoolGradV2Activation_conv2d_4	MaxPool_1Gradients/Conv2DBackpropInput_5Const_69Const_70*
paddingSAME*
T0*
data_formatNHWC
E
Gradients/Identity_21IdentityGradients/MaxPoolGradV2_3*
T0
K
Gradients/ReluGrad_8ReluGradGradients/Identity_21	BiasAdd_1*
T0
\
Gradients/BiasAddGrad_6BiasAddGradGradients/ReluGrad_8*
T0*
data_formatNHWC
@
Gradients/Identity_22IdentityGradients/ReluGrad_8*
T0
=
Gradients/Shape_22ShapeMaxPool*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_6Conv2DBackpropInputGradients/Shape_22conv2d_4_conv2d_kernelGradients/Identity_22*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
L
Gradients/Shape_23Shapeconv2d_4_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_6Conv2DBackpropFilterMaxPoolGradients/Shape_23Gradients/Identity_22*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
?
Gradients/MaxPoolGradV2_4MaxPoolGradV2Activation_conv2d_2MaxPoolGradients/Conv2DBackpropInput_6Const_67Const_68*
paddingSAME*
T0*
data_formatNHWC
E
Gradients/Identity_23IdentityGradients/MaxPoolGradV2_4*
T0
I
Gradients/ReluGrad_9ReluGradGradients/Identity_23BiasAdd*
T0
\
Gradients/BiasAddGrad_7BiasAddGradGradients/ReluGrad_9*
T0*
data_formatNHWC
@
Gradients/Identity_24IdentityGradients/ReluGrad_9*
T0
N
Gradients/Shape_24Shapedefault_data_placeholder*
T0*
out_type0
?
Gradients/Conv2DBackpropInput_7Conv2DBackpropInputGradients/Shape_24conv2d_2_conv2d_kernelGradients/Identity_24*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
L
Gradients/Shape_25Shapeconv2d_2_conv2d_kernel*
T0*
out_type0
?
 Gradients/Conv2DBackpropFilter_7Conv2DBackpropFilterdefault_data_placeholderGradients/Shape_25Gradients/Identity_24*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
?
ShapeShapeconv2d_2_conv2d_kernel*
T0*
out_type0
5
Const_78Const*
valueB
 *    *
dtype0
[
'Init_optimizer_conv2d_2_conv2d_kernel-mFillShapeConst_78*
T0*

index_type0
~
"optimizer_conv2d_2_conv2d_kernel-m
VariableV2*
shape: *
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_2_conv2d_kernel-mAssign"optimizer_conv2d_2_conv2d_kernel-m'Init_optimizer_conv2d_2_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_1Shapeconv2d_2_conv2d_kernel*
T0*
out_type0
5
Const_79Const*
valueB
 *    *
dtype0
]
'Init_optimizer_conv2d_2_conv2d_kernel-vFillShape_1Const_79*
T0*

index_type0
~
"optimizer_conv2d_2_conv2d_kernel-v
VariableV2*
shape: *
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_2_conv2d_kernel-vAssign"optimizer_conv2d_2_conv2d_kernel-v'Init_optimizer_conv2d_2_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
?
Shape_2Shapeconv2d_2_conv2d_bias*
T0*
out_type0
5
Const_80Const*
valueB
 *    *
dtype0
[
%Init_optimizer_conv2d_2_conv2d_bias-mFillShape_2Const_80*
T0*

index_type0
p
 optimizer_conv2d_2_conv2d_bias-m
VariableV2*
shape: *
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_2_conv2d_bias-mAssign optimizer_conv2d_2_conv2d_bias-m%Init_optimizer_conv2d_2_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
?
Shape_3Shapeconv2d_2_conv2d_bias*
T0*
out_type0
5
Const_81Const*
valueB
 *    *
dtype0
[
%Init_optimizer_conv2d_2_conv2d_bias-vFillShape_3Const_81*
T0*

index_type0
p
 optimizer_conv2d_2_conv2d_bias-v
VariableV2*
shape: *
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_2_conv2d_bias-vAssign optimizer_conv2d_2_conv2d_bias-v%Init_optimizer_conv2d_2_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
A
Shape_4Shapeconv2d_4_conv2d_kernel*
T0*
out_type0
5
Const_82Const*
valueB
 *    *
dtype0
]
'Init_optimizer_conv2d_4_conv2d_kernel-mFillShape_4Const_82*
T0*

index_type0
~
"optimizer_conv2d_4_conv2d_kernel-m
VariableV2*
shape: @*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_4_conv2d_kernel-mAssign"optimizer_conv2d_4_conv2d_kernel-m'Init_optimizer_conv2d_4_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_5Shapeconv2d_4_conv2d_kernel*
T0*
out_type0
5
Const_83Const*
valueB
 *    *
dtype0
]
'Init_optimizer_conv2d_4_conv2d_kernel-vFillShape_5Const_83*
T0*

index_type0
~
"optimizer_conv2d_4_conv2d_kernel-v
VariableV2*
shape: @*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_4_conv2d_kernel-vAssign"optimizer_conv2d_4_conv2d_kernel-v'Init_optimizer_conv2d_4_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
?
Shape_6Shapeconv2d_4_conv2d_bias*
T0*
out_type0
5
Const_84Const*
valueB
 *    *
dtype0
[
%Init_optimizer_conv2d_4_conv2d_bias-mFillShape_6Const_84*
T0*

index_type0
p
 optimizer_conv2d_4_conv2d_bias-m
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_4_conv2d_bias-mAssign optimizer_conv2d_4_conv2d_bias-m%Init_optimizer_conv2d_4_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
?
Shape_7Shapeconv2d_4_conv2d_bias*
T0*
out_type0
5
Const_85Const*
valueB
 *    *
dtype0
[
%Init_optimizer_conv2d_4_conv2d_bias-vFillShape_7Const_85*
T0*

index_type0
p
 optimizer_conv2d_4_conv2d_bias-v
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_4_conv2d_bias-vAssign optimizer_conv2d_4_conv2d_bias-v%Init_optimizer_conv2d_4_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
A
Shape_8Shapeconv2d_6_conv2d_kernel*
T0*
out_type0
5
Const_86Const*
valueB
 *    *
dtype0
]
'Init_optimizer_conv2d_6_conv2d_kernel-mFillShape_8Const_86*
T0*

index_type0

"optimizer_conv2d_6_conv2d_kernel-m
VariableV2*
shape:@?*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_6_conv2d_kernel-mAssign"optimizer_conv2d_6_conv2d_kernel-m'Init_optimizer_conv2d_6_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_9Shapeconv2d_6_conv2d_kernel*
T0*
out_type0
5
Const_87Const*
valueB
 *    *
dtype0
]
'Init_optimizer_conv2d_6_conv2d_kernel-vFillShape_9Const_87*
T0*

index_type0

"optimizer_conv2d_6_conv2d_kernel-v
VariableV2*
shape:@?*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_6_conv2d_kernel-vAssign"optimizer_conv2d_6_conv2d_kernel-v'Init_optimizer_conv2d_6_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
@
Shape_10Shapeconv2d_6_conv2d_bias*
T0*
out_type0
5
Const_88Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_6_conv2d_bias-mFillShape_10Const_88*
T0*

index_type0
q
 optimizer_conv2d_6_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_6_conv2d_bias-mAssign optimizer_conv2d_6_conv2d_bias-m%Init_optimizer_conv2d_6_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
@
Shape_11Shapeconv2d_6_conv2d_bias*
T0*
out_type0
5
Const_89Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_6_conv2d_bias-vFillShape_11Const_89*
T0*

index_type0
q
 optimizer_conv2d_6_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_6_conv2d_bias-vAssign optimizer_conv2d_6_conv2d_bias-v%Init_optimizer_conv2d_6_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
B
Shape_12Shapeconv2d_7_conv2d_kernel*
T0*
out_type0
5
Const_90Const*
valueB
 *    *
dtype0
^
'Init_optimizer_conv2d_7_conv2d_kernel-mFillShape_12Const_90*
T0*

index_type0
?
"optimizer_conv2d_7_conv2d_kernel-m
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_7_conv2d_kernel-mAssign"optimizer_conv2d_7_conv2d_kernel-m'Init_optimizer_conv2d_7_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
B
Shape_13Shapeconv2d_7_conv2d_kernel*
T0*
out_type0
5
Const_91Const*
valueB
 *    *
dtype0
^
'Init_optimizer_conv2d_7_conv2d_kernel-vFillShape_13Const_91*
T0*

index_type0
?
"optimizer_conv2d_7_conv2d_kernel-v
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_7_conv2d_kernel-vAssign"optimizer_conv2d_7_conv2d_kernel-v'Init_optimizer_conv2d_7_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
@
Shape_14Shapeconv2d_7_conv2d_bias*
T0*
out_type0
5
Const_92Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_7_conv2d_bias-mFillShape_14Const_92*
T0*

index_type0
q
 optimizer_conv2d_7_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_7_conv2d_bias-mAssign optimizer_conv2d_7_conv2d_bias-m%Init_optimizer_conv2d_7_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
@
Shape_15Shapeconv2d_7_conv2d_bias*
T0*
out_type0
5
Const_93Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_7_conv2d_bias-vFillShape_15Const_93*
T0*

index_type0
q
 optimizer_conv2d_7_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_7_conv2d_bias-vAssign optimizer_conv2d_7_conv2d_bias-v%Init_optimizer_conv2d_7_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
B
Shape_16Shapeconv2d_9_conv2d_kernel*
T0*
out_type0
5
Const_94Const*
valueB
 *    *
dtype0
^
'Init_optimizer_conv2d_9_conv2d_kernel-mFillShape_16Const_94*
T0*

index_type0
?
"optimizer_conv2d_9_conv2d_kernel-m
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_9_conv2d_kernel-mAssign"optimizer_conv2d_9_conv2d_kernel-m'Init_optimizer_conv2d_9_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
B
Shape_17Shapeconv2d_9_conv2d_kernel*
T0*
out_type0
5
Const_95Const*
valueB
 *    *
dtype0
^
'Init_optimizer_conv2d_9_conv2d_kernel-vFillShape_17Const_95*
T0*

index_type0
?
"optimizer_conv2d_9_conv2d_kernel-v
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
)Assign_optimizer_conv2d_9_conv2d_kernel-vAssign"optimizer_conv2d_9_conv2d_kernel-v'Init_optimizer_conv2d_9_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
@
Shape_18Shapeconv2d_9_conv2d_bias*
T0*
out_type0
5
Const_96Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_9_conv2d_bias-mFillShape_18Const_96*
T0*

index_type0
q
 optimizer_conv2d_9_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_9_conv2d_bias-mAssign optimizer_conv2d_9_conv2d_bias-m%Init_optimizer_conv2d_9_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
@
Shape_19Shapeconv2d_9_conv2d_bias*
T0*
out_type0
5
Const_97Const*
valueB
 *    *
dtype0
\
%Init_optimizer_conv2d_9_conv2d_bias-vFillShape_19Const_97*
T0*

index_type0
q
 optimizer_conv2d_9_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_conv2d_9_conv2d_bias-vAssign optimizer_conv2d_9_conv2d_bias-v%Init_optimizer_conv2d_9_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
C
Shape_20Shapeconv2d_10_conv2d_kernel*
T0*
out_type0
5
Const_98Const*
valueB
 *    *
dtype0
_
(Init_optimizer_conv2d_10_conv2d_kernel-mFillShape_20Const_98*
T0*

index_type0
?
#optimizer_conv2d_10_conv2d_kernel-m
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_10_conv2d_kernel-mAssign#optimizer_conv2d_10_conv2d_kernel-m(Init_optimizer_conv2d_10_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
C
Shape_21Shapeconv2d_10_conv2d_kernel*
T0*
out_type0
5
Const_99Const*
valueB
 *    *
dtype0
_
(Init_optimizer_conv2d_10_conv2d_kernel-vFillShape_21Const_99*
T0*

index_type0
?
#optimizer_conv2d_10_conv2d_kernel-v
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_10_conv2d_kernel-vAssign#optimizer_conv2d_10_conv2d_kernel-v(Init_optimizer_conv2d_10_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
A
Shape_22Shapeconv2d_10_conv2d_bias*
T0*
out_type0
6
	Const_100Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_10_conv2d_bias-mFillShape_22	Const_100*
T0*

index_type0
r
!optimizer_conv2d_10_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_10_conv2d_bias-mAssign!optimizer_conv2d_10_conv2d_bias-m&Init_optimizer_conv2d_10_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
A
Shape_23Shapeconv2d_10_conv2d_bias*
T0*
out_type0
6
	Const_101Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_10_conv2d_bias-vFillShape_23	Const_101*
T0*

index_type0
r
!optimizer_conv2d_10_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_10_conv2d_bias-vAssign!optimizer_conv2d_10_conv2d_bias-v&Init_optimizer_conv2d_10_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
C
Shape_24Shapeconv2d_12_conv2d_kernel*
T0*
out_type0
6
	Const_102Const*
valueB
 *    *
dtype0
`
(Init_optimizer_conv2d_12_conv2d_kernel-mFillShape_24	Const_102*
T0*

index_type0
?
#optimizer_conv2d_12_conv2d_kernel-m
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_12_conv2d_kernel-mAssign#optimizer_conv2d_12_conv2d_kernel-m(Init_optimizer_conv2d_12_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
C
Shape_25Shapeconv2d_12_conv2d_kernel*
T0*
out_type0
6
	Const_103Const*
valueB
 *    *
dtype0
`
(Init_optimizer_conv2d_12_conv2d_kernel-vFillShape_25	Const_103*
T0*

index_type0
?
#optimizer_conv2d_12_conv2d_kernel-v
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_12_conv2d_kernel-vAssign#optimizer_conv2d_12_conv2d_kernel-v(Init_optimizer_conv2d_12_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
A
Shape_26Shapeconv2d_12_conv2d_bias*
T0*
out_type0
6
	Const_104Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_12_conv2d_bias-mFillShape_26	Const_104*
T0*

index_type0
r
!optimizer_conv2d_12_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_12_conv2d_bias-mAssign!optimizer_conv2d_12_conv2d_bias-m&Init_optimizer_conv2d_12_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
A
Shape_27Shapeconv2d_12_conv2d_bias*
T0*
out_type0
6
	Const_105Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_12_conv2d_bias-vFillShape_27	Const_105*
T0*

index_type0
r
!optimizer_conv2d_12_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_12_conv2d_bias-vAssign!optimizer_conv2d_12_conv2d_bias-v&Init_optimizer_conv2d_12_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
C
Shape_28Shapeconv2d_13_conv2d_kernel*
T0*
out_type0
6
	Const_106Const*
valueB
 *    *
dtype0
`
(Init_optimizer_conv2d_13_conv2d_kernel-mFillShape_28	Const_106*
T0*

index_type0
?
#optimizer_conv2d_13_conv2d_kernel-m
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_13_conv2d_kernel-mAssign#optimizer_conv2d_13_conv2d_kernel-m(Init_optimizer_conv2d_13_conv2d_kernel-m*
use_locking(*
T0*
validate_shape(
C
Shape_29Shapeconv2d_13_conv2d_kernel*
T0*
out_type0
6
	Const_107Const*
valueB
 *    *
dtype0
`
(Init_optimizer_conv2d_13_conv2d_kernel-vFillShape_29	Const_107*
T0*

index_type0
?
#optimizer_conv2d_13_conv2d_kernel-v
VariableV2*
shape:??*
shared_name *
dtype0*
	container 
?
*Assign_optimizer_conv2d_13_conv2d_kernel-vAssign#optimizer_conv2d_13_conv2d_kernel-v(Init_optimizer_conv2d_13_conv2d_kernel-v*
use_locking(*
T0*
validate_shape(
A
Shape_30Shapeconv2d_13_conv2d_bias*
T0*
out_type0
6
	Const_108Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_13_conv2d_bias-mFillShape_30	Const_108*
T0*

index_type0
r
!optimizer_conv2d_13_conv2d_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_13_conv2d_bias-mAssign!optimizer_conv2d_13_conv2d_bias-m&Init_optimizer_conv2d_13_conv2d_bias-m*
use_locking(*
T0*
validate_shape(
A
Shape_31Shapeconv2d_13_conv2d_bias*
T0*
out_type0
6
	Const_109Const*
valueB
 *    *
dtype0
^
&Init_optimizer_conv2d_13_conv2d_bias-vFillShape_31	Const_109*
T0*

index_type0
r
!optimizer_conv2d_13_conv2d_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_conv2d_13_conv2d_bias-vAssign!optimizer_conv2d_13_conv2d_bias-v&Init_optimizer_conv2d_13_conv2d_bias-v*
use_locking(*
T0*
validate_shape(
A
Shape_32Shapedense_16_dense_kernel*
T0*
out_type0
6
	Const_110Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_16_dense_kernel-mFillShape_32	Const_110*
T0*

index_type0
w
!optimizer_dense_16_dense_kernel-m
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_16_dense_kernel-mAssign!optimizer_dense_16_dense_kernel-m&Init_optimizer_dense_16_dense_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_33Shapedense_16_dense_kernel*
T0*
out_type0
6
	Const_111Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_16_dense_kernel-vFillShape_33	Const_111*
T0*

index_type0
w
!optimizer_dense_16_dense_kernel-v
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_16_dense_kernel-vAssign!optimizer_dense_16_dense_kernel-v&Init_optimizer_dense_16_dense_kernel-v*
use_locking(*
T0*
validate_shape(
?
Shape_34Shapedense_16_dense_bias*
T0*
out_type0
6
	Const_112Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_16_dense_bias-mFillShape_34	Const_112*
T0*

index_type0
p
optimizer_dense_16_dense_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_16_dense_bias-mAssignoptimizer_dense_16_dense_bias-m$Init_optimizer_dense_16_dense_bias-m*
use_locking(*
T0*
validate_shape(
?
Shape_35Shapedense_16_dense_bias*
T0*
out_type0
6
	Const_113Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_16_dense_bias-vFillShape_35	Const_113*
T0*

index_type0
p
optimizer_dense_16_dense_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_16_dense_bias-vAssignoptimizer_dense_16_dense_bias-v$Init_optimizer_dense_16_dense_bias-v*
use_locking(*
T0*
validate_shape(
A
Shape_36Shapedense_17_dense_kernel*
T0*
out_type0
6
	Const_114Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_17_dense_kernel-mFillShape_36	Const_114*
T0*

index_type0
w
!optimizer_dense_17_dense_kernel-m
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_17_dense_kernel-mAssign!optimizer_dense_17_dense_kernel-m&Init_optimizer_dense_17_dense_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_37Shapedense_17_dense_kernel*
T0*
out_type0
6
	Const_115Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_17_dense_kernel-vFillShape_37	Const_115*
T0*

index_type0
w
!optimizer_dense_17_dense_kernel-v
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_17_dense_kernel-vAssign!optimizer_dense_17_dense_kernel-v&Init_optimizer_dense_17_dense_kernel-v*
use_locking(*
T0*
validate_shape(
?
Shape_38Shapedense_17_dense_bias*
T0*
out_type0
6
	Const_116Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_17_dense_bias-mFillShape_38	Const_116*
T0*

index_type0
p
optimizer_dense_17_dense_bias-m
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_17_dense_bias-mAssignoptimizer_dense_17_dense_bias-m$Init_optimizer_dense_17_dense_bias-m*
use_locking(*
T0*
validate_shape(
?
Shape_39Shapedense_17_dense_bias*
T0*
out_type0
6
	Const_117Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_17_dense_bias-vFillShape_39	Const_117*
T0*

index_type0
p
optimizer_dense_17_dense_bias-v
VariableV2*
shape:?*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_17_dense_bias-vAssignoptimizer_dense_17_dense_bias-v$Init_optimizer_dense_17_dense_bias-v*
use_locking(*
T0*
validate_shape(
A
Shape_40Shapedense_18_dense_kernel*
T0*
out_type0
6
	Const_118Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_18_dense_kernel-mFillShape_40	Const_118*
T0*

index_type0
v
!optimizer_dense_18_dense_kernel-m
VariableV2*
shape:	?
*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_18_dense_kernel-mAssign!optimizer_dense_18_dense_kernel-m&Init_optimizer_dense_18_dense_kernel-m*
use_locking(*
T0*
validate_shape(
A
Shape_41Shapedense_18_dense_kernel*
T0*
out_type0
6
	Const_119Const*
valueB
 *    *
dtype0
^
&Init_optimizer_dense_18_dense_kernel-vFillShape_41	Const_119*
T0*

index_type0
v
!optimizer_dense_18_dense_kernel-v
VariableV2*
shape:	?
*
shared_name *
dtype0*
	container 
?
(Assign_optimizer_dense_18_dense_kernel-vAssign!optimizer_dense_18_dense_kernel-v&Init_optimizer_dense_18_dense_kernel-v*
use_locking(*
T0*
validate_shape(
?
Shape_42Shapedense_18_dense_bias*
T0*
out_type0
6
	Const_120Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_18_dense_bias-mFillShape_42	Const_120*
T0*

index_type0
o
optimizer_dense_18_dense_bias-m
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_18_dense_bias-mAssignoptimizer_dense_18_dense_bias-m$Init_optimizer_dense_18_dense_bias-m*
use_locking(*
T0*
validate_shape(
?
Shape_43Shapedense_18_dense_bias*
T0*
out_type0
6
	Const_121Const*
valueB
 *    *
dtype0
\
$Init_optimizer_dense_18_dense_bias-vFillShape_43	Const_121*
T0*

index_type0
o
optimizer_dense_18_dense_bias-v
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
?
&Assign_optimizer_dense_18_dense_bias-vAssignoptimizer_dense_18_dense_bias-v$Init_optimizer_dense_18_dense_bias-v*
use_locking(*
T0*
validate_shape(
a
optimizer_beta1_power
VariableV2*
shape: *
shared_name *
dtype0*
	container 
G
Init_optimizer_beta1_powerConst*
valueB
 *fff?*
dtype0
?
Assign_optimizer_beta1_powerAssignoptimizer_beta1_powerInit_optimizer_beta1_power*
use_locking(*
T0*
validate_shape(
a
optimizer_beta2_power
VariableV2*
shape: *
shared_name *
dtype0*
	container 
G
Init_optimizer_beta2_powerConst*
valueB
 *w??*
dtype0
?
Assign_optimizer_beta2_powerAssignoptimizer_beta2_powerInit_optimizer_beta2_power*
use_locking(*
T0*
validate_shape(
6
	Const_122Const*
valueB
 *fff?*
dtype0
6
	Const_123Const*
valueB
 *w??*
dtype0
6
	Const_124Const*
valueB
 *o?:*
dtype0
6
	Const_125Const*
valueB
 *???3*
dtype0
?
	ApplyAdam	ApplyAdamconv2d_2_conv2d_kernel"optimizer_conv2d_2_conv2d_kernel-m"optimizer_conv2d_2_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_7*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_1	ApplyAdamconv2d_2_conv2d_bias optimizer_conv2d_2_conv2d_bias-m optimizer_conv2d_2_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_7*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_2	ApplyAdamconv2d_4_conv2d_kernel"optimizer_conv2d_4_conv2d_kernel-m"optimizer_conv2d_4_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_6*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_3	ApplyAdamconv2d_4_conv2d_bias optimizer_conv2d_4_conv2d_bias-m optimizer_conv2d_4_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_6*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_4	ApplyAdamconv2d_6_conv2d_kernel"optimizer_conv2d_6_conv2d_kernel-m"optimizer_conv2d_6_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_5*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_5	ApplyAdamconv2d_6_conv2d_bias optimizer_conv2d_6_conv2d_bias-m optimizer_conv2d_6_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_5*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_6	ApplyAdamconv2d_7_conv2d_kernel"optimizer_conv2d_7_conv2d_kernel-m"optimizer_conv2d_7_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_4*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_7	ApplyAdamconv2d_7_conv2d_bias optimizer_conv2d_7_conv2d_bias-m optimizer_conv2d_7_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_4*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_8	ApplyAdamconv2d_9_conv2d_kernel"optimizer_conv2d_9_conv2d_kernel-m"optimizer_conv2d_9_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_3*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_9	ApplyAdamconv2d_9_conv2d_bias optimizer_conv2d_9_conv2d_bias-m optimizer_conv2d_9_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_3*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_10	ApplyAdamconv2d_10_conv2d_kernel#optimizer_conv2d_10_conv2d_kernel-m#optimizer_conv2d_10_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_2*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_11	ApplyAdamconv2d_10_conv2d_bias!optimizer_conv2d_10_conv2d_bias-m!optimizer_conv2d_10_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_2*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_12	ApplyAdamconv2d_12_conv2d_kernel#optimizer_conv2d_12_conv2d_kernel-m#optimizer_conv2d_12_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125 Gradients/Conv2DBackpropFilter_1*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_13	ApplyAdamconv2d_12_conv2d_bias!optimizer_conv2d_12_conv2d_bias-m!optimizer_conv2d_12_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad_1*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_14	ApplyAdamconv2d_13_conv2d_kernel#optimizer_conv2d_13_conv2d_kernel-m#optimizer_conv2d_13_conv2d_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/Conv2DBackpropFilter*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_15	ApplyAdamconv2d_13_conv2d_bias!optimizer_conv2d_13_conv2d_bias-m!optimizer_conv2d_13_conv2d_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/BiasAddGrad*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_16	ApplyAdamdense_16_dense_kernel!optimizer_dense_16_dense_kernel-m!optimizer_dense_16_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/MatMul_5*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_17	ApplyAdamdense_16_dense_biasoptimizer_dense_16_dense_bias-moptimizer_dense_16_dense_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/Reshape_6*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_18	ApplyAdamdense_17_dense_kernel!optimizer_dense_17_dense_kernel-m!optimizer_dense_17_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/MatMul_3*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_19	ApplyAdamdense_17_dense_biasoptimizer_dense_17_dense_bias-moptimizer_dense_17_dense_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/Reshape_4*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_20	ApplyAdamdense_18_dense_kernel!optimizer_dense_18_dense_kernel-m!optimizer_dense_18_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/MatMul_1*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_21	ApplyAdamdense_18_dense_biasoptimizer_dense_18_dense_bias-moptimizer_dense_18_dense_bias-voptimizer_beta1_poweroptimizer_beta2_power	Const_124	Const_122	Const_123	Const_125Gradients/Reshape_2*
use_locking( *
T0*
use_nesterov( 
5
MulMuloptimizer_beta1_power	Const_122*
T0
^
AssignAssignoptimizer_beta1_powerMul*
use_locking(*
T0*
validate_shape(
7
Mul_1Muloptimizer_beta2_power	Const_123*
T0
b
Assign_1Assignoptimizer_beta2_powerMul_1*
use_locking(*
T0*
validate_shape(
7
default_outputSoftmaxActivation_dense_18*
T0
3
	Const_126Const*
value	B :*
dtype0
S
ArgMaxArgMaxdefault_output	Const_126*

Tidx0*
T0*
output_type0	
3
	Const_127Const*
value	B :*
dtype0
R
ArgMax_1ArgMaxPlaceholder	Const_127*

Tidx0*
T0*
output_type0	
I
EqualEqualArgMaxArgMax_1*
incompatible_shape_error(*
T0	
>
Cast_22CastEqual*

SrcT0
*
Truncate( *

DstT0
3
	Const_128Const*
value	B : *
dtype0
F
MeanMeanCast_22	Const_128*

Tidx0*
	keep_dims( *
T0 "?
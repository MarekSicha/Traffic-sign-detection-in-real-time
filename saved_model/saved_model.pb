
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02unknown8­	

conv2d_4_1/kernelVarHandleOp*
_output_shapes
: *"
shared_nameconv2d_4_1/kernel*
dtype0*
shape: 

%conv2d_4_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_4_1/kernel*&
_output_shapes
: *
dtype0
v
conv2d_4_1/biasVarHandleOp*
_output_shapes
: * 
shared_nameconv2d_4_1/bias*
shape: *
dtype0
o
#conv2d_4_1/bias/Read/ReadVariableOpReadVariableOpconv2d_4_1/bias*
_output_shapes
: *
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: * 
shared_nameconv2d_5/kernel*
shape:  *
dtype0
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_5/biasVarHandleOp*
shape: *
_output_shapes
: *
shared_nameconv2d_5/bias*
dtype0
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0

conv2d_6/kernelVarHandleOp*
shape: @*
_output_shapes
: *
dtype0* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_6/biasVarHandleOp*
shape:@*
dtype0*
shared_nameconv2d_6/bias*
_output_shapes
: 
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0

conv2d_7/kernelVarHandleOp*
shape:@@*
dtype0* 
shared_nameconv2d_7/kernel*
_output_shapes
: 
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
~
dense_2_1/kernelVarHandleOp*
shape:
*
dtype0*!
shared_namedense_2_1/kernel*
_output_shapes
: 
w
$dense_2_1/kernel/Read/ReadVariableOpReadVariableOpdense_2_1/kernel*
dtype0* 
_output_shapes
:

u
dense_2_1/biasVarHandleOp*
shape:*
_output_shapes
: *
shared_namedense_2_1/bias*
dtype0
n
"dense_2_1/bias/Read/ReadVariableOpReadVariableOpdense_2_1/bias*
_output_shapes	
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
shared_namedense_3/kernel*
dtype0*
shape:	+
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	+*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
shared_namedense_3/bias*
dtype0*
shape:+
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:+*
dtype0
f
	Adam/iterVarHandleOp*
shape: *
_output_shapes
: *
dtype0	*
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
dtype0*
shared_nameAdam/beta_1*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
shape: *
shared_nameAdam/beta_2*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
dtype0*
shape: *
shared_name
Adam/decay*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *
_output_shapes
: *#
shared_nameAdam/learning_rate*
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
_output_shapes
: *
shared_nametotal*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
_output_shapes
: *
shared_namecount*
dtype0*
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv2d_4_1/kernel/mVarHandleOp*
shape: *
dtype0*)
shared_nameAdam/conv2d_4_1/kernel/m*
_output_shapes
: 

,Adam/conv2d_4_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/kernel/m*
dtype0*&
_output_shapes
: 

Adam/conv2d_4_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*'
shared_nameAdam/conv2d_4_1/bias/m*
shape: 
}
*Adam/conv2d_4_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/bias/m*
dtype0*
_output_shapes
: 

Adam/conv2d_5/kernel/mVarHandleOp*
shape:  *
dtype0*'
shared_nameAdam/conv2d_5/kernel/m*
_output_shapes
: 

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
shape: *
dtype0*%
shared_nameAdam/conv2d_5/bias/m*
_output_shapes
: 
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_6/kernel/mVarHandleOp*
shape: @*
dtype0*'
shared_nameAdam/conv2d_6/kernel/m*
_output_shapes
: 

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*
dtype0*&
_output_shapes
: @

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_6/bias/m*
dtype0*
shape:@
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
dtype0*
_output_shapes
:@

Adam/conv2d_7/kernel/mVarHandleOp*
shape:@@*
_output_shapes
: *
dtype0*'
shared_nameAdam/conv2d_7/kernel/m

*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_7/bias/m*
dtype0*
shape:@
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_2_1/kernel/mVarHandleOp*
shape:
*
dtype0*(
shared_nameAdam/dense_2_1/kernel/m*
_output_shapes
: 

+Adam/dense_2_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2_1/kernel/m*
dtype0* 
_output_shapes
:


Adam/dense_2_1/bias/mVarHandleOp*
shape:*
dtype0*&
shared_nameAdam/dense_2_1/bias/m*
_output_shapes
: 
|
)Adam/dense_2_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2_1/bias/m*
dtype0*
_output_shapes	
:

Adam/dense_3/kernel/mVarHandleOp*
shape:	+*
dtype0*&
shared_nameAdam/dense_3/kernel/m*
_output_shapes
: 

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	+*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *$
shared_nameAdam/dense_3/bias/m*
dtype0*
shape:+
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:+*
dtype0

Adam/conv2d_4_1/kernel/vVarHandleOp*
shape: *
_output_shapes
: *
dtype0*)
shared_nameAdam/conv2d_4_1/kernel/v

,Adam/conv2d_4_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_4_1/bias/vVarHandleOp*
shape: *
_output_shapes
: *'
shared_nameAdam/conv2d_4_1/bias/v*
dtype0
}
*Adam/conv2d_4_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
dtype0*
shape:  *'
shared_nameAdam/conv2d_5/kernel/v*
_output_shapes
: 

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
shape: *
dtype0*%
shared_nameAdam/conv2d_5/bias/v*
_output_shapes
: 
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_6/kernel/vVarHandleOp*
shape: @*
dtype0*'
shared_nameAdam/conv2d_6/kernel/v*
_output_shapes
: 

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*
dtype0*&
_output_shapes
: @

Adam/conv2d_6/bias/vVarHandleOp*
shape:@*
_output_shapes
: *%
shared_nameAdam/conv2d_6/bias/v*
dtype0
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
dtype0*
_output_shapes
:@

Adam/conv2d_7/kernel/vVarHandleOp*
shape:@@*
_output_shapes
: *'
shared_nameAdam/conv2d_7/kernel/v*
dtype0

*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_7/bias/vVarHandleOp*
shape:@*
dtype0*%
shared_nameAdam/conv2d_7/bias/v*
_output_shapes
: 
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_2_1/kernel/vVarHandleOp*
dtype0*(
shared_nameAdam/dense_2_1/kernel/v*
_output_shapes
: *
shape:


+Adam/dense_2_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_2_1/bias/vVarHandleOp*
shape:*
dtype0*&
shared_nameAdam/dense_2_1/bias/v*
_output_shapes
: 
|
)Adam/dense_2_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2_1/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
shape:	+*
dtype0*&
shared_nameAdam/dense_3/kernel/v*
_output_shapes
: 

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
dtype0*
_output_shapes
:	+
~
Adam/dense_3/bias/vVarHandleOp*
dtype0*$
shared_nameAdam/dense_3/bias/v*
_output_shapes
: *
shape:+
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
¸K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*óJ
valueéJBæJ BßJ
«
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
R
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
°
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemmmm'm(m-m.m?m@mImJmv v¡v¢v£'v¤(v¥-v¦.v§?v¨@v©IvªJv«
 
V
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11
V
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11

Tnon_trainable_variables

Ulayers
regularization_losses
	variables
Vmetrics
trainable_variables
Wlayer_regularization_losses
 
 
 
 

Xnon_trainable_variables

Ylayers
regularization_losses
	variables
Zmetrics
trainable_variables
[layer_regularization_losses
][
VARIABLE_VALUEconv2d_4_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_4_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

\non_trainable_variables

]layers
regularization_losses
	variables
^metrics
trainable_variables
_layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

`non_trainable_variables

alayers
regularization_losses
 	variables
bmetrics
!trainable_variables
clayer_regularization_losses
 
 
 

dnon_trainable_variables

elayers
#regularization_losses
$	variables
fmetrics
%trainable_variables
glayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1

hnon_trainable_variables

ilayers
)regularization_losses
*	variables
jmetrics
+trainable_variables
klayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1

lnon_trainable_variables

mlayers
/regularization_losses
0	variables
nmetrics
1trainable_variables
olayer_regularization_losses
 
 
 

pnon_trainable_variables

qlayers
3regularization_losses
4	variables
rmetrics
5trainable_variables
slayer_regularization_losses
 
 
 

tnon_trainable_variables

ulayers
7regularization_losses
8	variables
vmetrics
9trainable_variables
wlayer_regularization_losses
 
 
 

xnon_trainable_variables

ylayers
;regularization_losses
<	variables
zmetrics
=trainable_variables
{layer_regularization_losses
\Z
VARIABLE_VALUEdense_2_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_2_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1

|non_trainable_variables

}layers
Aregularization_losses
B	variables
~metrics
Ctrainable_variables
layer_regularization_losses
 
 
 

non_trainable_variables
layers
Eregularization_losses
F	variables
metrics
Gtrainable_variables
 layer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1

non_trainable_variables
layers
Kregularization_losses
L	variables
metrics
Mtrainable_variables
 layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
	7

8
9
10

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


total

count

_fn_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
¡
non_trainable_variables
layers
regularization_losses
	variables
metrics
trainable_variables
 layer_regularization_losses

0
1
 
 
 
~
VARIABLE_VALUEAdam/conv2d_4_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_4_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_2_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_4_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_4_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_2_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_4_inputPlaceholder*$
shape:ÿÿÿÿÿÿÿÿÿ  */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0
õ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_4_inputconv2d_4_1/kernelconv2d_4_1/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_2_1/kerneldense_2_1/biasdense_3/kerneldense_3/bias*
Tin
2*-
_gradient_op_typePartitionedCall-361499*-
f(R&
$__inference_signature_wrapper_361122*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
Tout
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
×
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_4_1/kernel/Read/ReadVariableOp#conv2d_4_1/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp$dense_2_1/kernel/Read/ReadVariableOp"dense_2_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_4_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_4_1/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp+Adam/dense_2_1/kernel/m/Read/ReadVariableOp)Adam/dense_2_1/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp,Adam/conv2d_4_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_4_1/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp+Adam/dense_2_1/kernel/v/Read/ReadVariableOp)Adam/dense_2_1/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*
_output_shapes
: *-
_gradient_op_typePartitionedCall-361564*(
f#R!
__inference__traced_save_361563*8
Tin1
/2-	*-
config_proto

CPU

GPU2*0J 8*
Tout
2
ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4_1/kernelconv2d_4_1/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_2_1/kerneldense_2_1/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_4_1/kernel/mAdam/conv2d_4_1/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense_2_1/kernel/mAdam/dense_2_1/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d_4_1/kernel/vAdam/conv2d_4_1/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense_2_1/kernel/vAdam/dense_2_1/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*7
Tin0
.2,*-
_gradient_op_typePartitionedCall-361706*+
f&R$
"__inference__traced_restore_361705*-
config_proto

CPU

GPU2*0J 8*
_output_shapes
: *
Tout
2âî

Ý
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpª
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
strides
*
T0*
paddingVALID 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0¥
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
©
ª
)__inference_conv2d_4_layer_call_fn_360680

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*-
_gradient_op_typePartitionedCall-360675*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669*-
config_proto

CPU

GPU2*0J 8*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Ú/
Ï
H__inference_sequential_1_layer_call_and_return_conditional_losses_361079

inputs+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2+
'conv2d_7_statefulpartitionedcall_args_1+
'conv2d_7_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669*-
_gradient_op_typePartitionedCall-360675*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tout
2µ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360700ß
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tin
2*
Tout
2*-
_gradient_op_typePartitionedCall-360719*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713´
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*-
_gradient_op_typePartitionedCall-360742*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
Tout
2µ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0'conv2d_7_statefulpartitionedcall_args_1'conv2d_7_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360767ß
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780*-
_gradient_op_typePartitionedCall-360786*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8Ò
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*-
_gradient_op_typePartitionedCall-360848*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360836*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8Å
flatten_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_gradient_op_typePartitionedCall-360865*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8£
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360889*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_360883Ë
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360927*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360939¢
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
Tin
2*
Tout
2*-
_gradient_op_typePartitionedCall-360961*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_360955À
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 
ü
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_361333

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
ç`

H__inference_sequential_1_layer_call_and_return_conditional_losses_361206

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¼
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
: *
dtype0¬
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
T0*
paddingVALID²
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: 
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
T0j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  Á
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
T0*
strides
*
paddingVALID²
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
T0¬
max_pooling2d_2/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
strides
*
ksize
*
paddingVALID¼
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @Æ
conv2d_6/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
T0*
paddingVALID²
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
T0j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
T0¼
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Á
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
strides
*
T0*
paddingVALID²
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0¬
max_pooling2d_3/MaxPoolMaxPoolconv2d_7/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
strides
*
ksize
*
paddingVALID[
dropout_2/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: g
dropout_2/dropout/ShapeShape max_pooling2d_3/MaxPool:output:0*
_output_shapes
:*
T0i
$dropout_2/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  ?*
_output_shapes
: *
dtype0¨
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
T0ª
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0È
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
dropout_2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_2/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: ¯
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_2/dropout/mulMul max_pooling2d_3/MaxPool:output:0dropout_2/dropout/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

DstT0
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
flatten_1/Reshape/shapeConst*
valueB"ÿÿÿÿ   *
_output_shapes
:*
dtype0
flatten_1/ReshapeReshapedropout_2/dropout/mul_1:z:0 flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0a
dense_2/ReluReludense_2/BiasAdd:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0[
dropout_3/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: a
dropout_3/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_3/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0i
$dropout_3/dropout/random_uniform/maxConst*
valueB
 *  ?*
_output_shapes
: *
dtype0¡
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0ª
$dropout_3/dropout/random_uniform/subSub-dropout_3/dropout/random_uniform/max:output:0-dropout_3/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Á
$dropout_3/dropout/random_uniform/mulMul7dropout_3/dropout/random_uniform/RandomUniform:output:0(dropout_3/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 dropout_3/dropout/random_uniformAdd(dropout_3/dropout/random_uniform/mul:z:0-dropout_3/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_3/dropout/subSub dropout_3/dropout/sub/x:output:0dropout_3/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_3/dropout/truediv/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
dropout_3/dropout/truedivRealDiv$dropout_3/dropout/truediv/x:output:0dropout_3/dropout/sub:z:0*
T0*
_output_shapes
: ¨
dropout_3/dropout/GreaterEqualGreaterEqual$dropout_3/dropout/random_uniform:z:0dropout_3/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/mulMuldense_2/Relu:activations:0dropout_3/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

DstT0*

SrcT0

dropout_3/dropout/mul_1Muldropout_3/dropout/mul:z:0dropout_3/dropout/Cast:y:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0³
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	+
dense_3/MatMulMatMuldropout_3/dropout/mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+°
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:+
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0ï
IdentityIdentitydense_3/Softmax:softmax:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 
Ä
F
*__inference_flatten_1_layer_call_fn_361338

inputs
identity
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-360865*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
²
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_361376

inputs
identityQ
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0£
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
²
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_360920

inputs
identityQ
dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: £
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

SrcT0
*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ö
c
*__inference_dropout_2_layer_call_fn_361322

inputs
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tin
2*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360829*
Tout
2*-
_gradient_op_typePartitionedCall-360840
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ú
©
(__inference_dense_3_layer_call_fn_361409

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_360955*-
config_proto

CPU

GPU2*0J 8*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*-
_gradient_op_typePartitionedCall-360961
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
×	
Ü
C__inference_dense_2_layer_call_and_return_conditional_losses_360883

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713

inputs
identity¢
MaxPoolMaxPoolinputs*
paddingVALID*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
strides
*
ksize
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_361317

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

Ý
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpª
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
: @*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
strides
*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0¥
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ï

$__inference_signature_wrapper_361122
conv2d_4_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2**
f%R#
!__inference__wrapped_model_360655*-
config_proto

CPU

GPU2*0J 8*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*-
_gradient_op_typePartitionedCall-361107
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :. *
(
_user_specified_nameconv2d_4_input: : : :
 : : : : :	 : 
¤
Ì
"__inference__traced_restore_361705
file_prefix&
"assignvariableop_conv2d_4_1_kernel&
"assignvariableop_1_conv2d_4_1_bias&
"assignvariableop_2_conv2d_5_kernel$
 assignvariableop_3_conv2d_5_bias&
"assignvariableop_4_conv2d_6_kernel$
 assignvariableop_5_conv2d_6_bias&
"assignvariableop_6_conv2d_7_kernel$
 assignvariableop_7_conv2d_7_bias'
#assignvariableop_8_dense_2_1_kernel%
!assignvariableop_9_dense_2_1_bias&
"assignvariableop_10_dense_3_kernel$
 assignvariableop_11_dense_3_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count0
,assignvariableop_19_adam_conv2d_4_1_kernel_m.
*assignvariableop_20_adam_conv2d_4_1_bias_m.
*assignvariableop_21_adam_conv2d_5_kernel_m,
(assignvariableop_22_adam_conv2d_5_bias_m.
*assignvariableop_23_adam_conv2d_6_kernel_m,
(assignvariableop_24_adam_conv2d_6_bias_m.
*assignvariableop_25_adam_conv2d_7_kernel_m,
(assignvariableop_26_adam_conv2d_7_bias_m/
+assignvariableop_27_adam_dense_2_1_kernel_m-
)assignvariableop_28_adam_dense_2_1_bias_m-
)assignvariableop_29_adam_dense_3_kernel_m+
'assignvariableop_30_adam_dense_3_bias_m0
,assignvariableop_31_adam_conv2d_4_1_kernel_v.
*assignvariableop_32_adam_conv2d_4_1_bias_v.
*assignvariableop_33_adam_conv2d_5_kernel_v,
(assignvariableop_34_adam_conv2d_5_bias_v.
*assignvariableop_35_adam_conv2d_6_kernel_v,
(assignvariableop_36_adam_conv2d_6_bias_v.
*assignvariableop_37_adam_conv2d_7_kernel_v,
(assignvariableop_38_adam_conv2d_7_bias_v/
+assignvariableop_39_adam_dense_2_1_kernel_v-
)assignvariableop_40_adam_dense_2_1_bias_v-
)assignvariableop_41_adam_dense_3_kernel_v+
'assignvariableop_42_adam_dense_3_bias_v
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*Â
value¸Bµ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:+Æ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0ø
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:~
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_4_1_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_4_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_6_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_6_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_2_1_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_2_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_3_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_3_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0*
dtype0	*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:{
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0{
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_4_1_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_4_1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_5_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_5_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_6_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_6_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_7_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_7_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype0P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_2_1_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_2_1_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_3_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_3_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_4_1_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_4_1_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_5_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_5_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_6_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_6_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_7_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype0P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_7_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_2_1_kernel_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_2_1_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype0P
Identity_41IdentityRestoreV2:tensors:41*
_output_shapes
:*
T0
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_3_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype0P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_3_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype0
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_44Identity_44:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:$ : : :  : : : : :( : :
 : :' : : :# : : : :	 :+ : : : :& : :+ '
%
_user_specified_namefile_prefix:" : : : : :* :% : : :! : : : : :) : : 
°
L
0__inference_max_pooling2d_3_layer_call_fn_360789

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tout
2*-
_gradient_op_typePartitionedCall-360786*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
Tin
2
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
½
F
*__inference_dropout_3_layer_call_fn_361391

inputs
identity
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360939*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360927a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ò/
×
H__inference_sequential_1_layer_call_and_return_conditional_losses_361002
conv2d_4_input+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2+
'conv2d_7_statefulpartitionedcall_args_1+
'conv2d_7_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_input'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360675*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669µ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *-
_gradient_op_typePartitionedCall-360700*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694ß
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tout
2*-
_gradient_op_typePartitionedCall-360719*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tin
2´
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360742*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8µ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0'conv2d_7_statefulpartitionedcall_args_1'conv2d_7_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360767*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8ß
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*-
_gradient_op_typePartitionedCall-360786*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tin
2Ò
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360848*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360836Å
flatten_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_gradient_op_typePartitionedCall-360865*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859£
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_gradient_op_typePartitionedCall-360889*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_360883Ë
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*-
_gradient_op_typePartitionedCall-360939*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360927*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tin
2¢
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360961*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_360955*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8À
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall: : :. *
(
_user_specified_nameconv2d_4_input: : : :
 : : : : :	 : 
×	
Ü
C__inference_dense_2_layer_call_and_return_conditional_losses_361349

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ü
©
(__inference_dense_2_layer_call_fn_361356

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360889*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_360883
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_360836

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
©
ª
)__inference_conv2d_7_layer_call_fn_360772

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*-
_gradient_op_typePartitionedCall-360767*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Á
c
*__inference_dropout_3_layer_call_fn_361386

inputs
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360931*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360920
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs


-__inference_sequential_1_layer_call_fn_361292

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tout
2*-
_gradient_op_typePartitionedCall-361080*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_361079*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : :
 : : : : :	 : 
æ2

H__inference_sequential_1_layer_call_and_return_conditional_losses_360973
conv2d_4_input+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2+
'conv2d_7_statefulpartitionedcall_args_1+
'conv2d_7_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_input'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360675*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ µ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360700*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694ß
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360719*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713´
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*-
config_proto

CPU

GPU2*0J 8*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*-
_gradient_op_typePartitionedCall-360742*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736µ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0'conv2d_7_statefulpartitionedcall_args_1'conv2d_7_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360767*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8ß
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360786*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8â
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360840*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360829*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8Í
flatten_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360865*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8£
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360889*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_360883*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8ÿ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tout
2*-
config_proto

CPU

GPU2*0J 8*-
_gradient_op_typePartitionedCall-360931*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360920*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tin
2ª
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360961*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_360955*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall: : :. *
(
_user_specified_nameconv2d_4_input: : : :
 : : : : :	 : 

Ý
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpª
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  ¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0*
paddingVALID 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0¥
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ò
F
*__inference_dropout_2_layer_call_fn_361327

inputs
identity¦
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-360848*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360836h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

¡
-__inference_sequential_1_layer_call_fn_361095
conv2d_4_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*-
_gradient_op_typePartitionedCall-361080*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_361079*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :. *
(
_user_specified_nameconv2d_4_input: : : :
 : : : : :	 : 
©
ª
)__inference_conv2d_6_layer_call_fn_360747

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736*-
_gradient_op_typePartitionedCall-360742*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 


-__inference_sequential_1_layer_call_fn_361275

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_361032*-
_gradient_op_typePartitionedCall-361033*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : :
 : : : : :	 : 
¯R
þ
__inference__traced_save_361563
file_prefix0
,savev2_conv2d_4_1_kernel_read_readvariableop.
*savev2_conv2d_4_1_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop/
+savev2_dense_2_1_kernel_read_readvariableop-
)savev2_dense_2_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_4_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_4_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop6
2savev2_adam_dense_2_1_kernel_m_read_readvariableop4
0savev2_adam_dense_2_1_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop7
3savev2_adam_conv2d_4_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_4_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop6
2savev2_adam_dense_2_1_kernel_v_read_readvariableop4
0savev2_adam_dense_2_1_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_b211f14c07a844de879f210b6c4e07e9/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*Â
value¸Bµ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:+*
dtype0Ã
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_4_1_kernel_read_readvariableop*savev2_conv2d_4_1_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop+savev2_dense_2_1_kernel_read_readvariableop)savev2_dense_2_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_4_1_kernel_m_read_readvariableop1savev2_adam_conv2d_4_1_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop2savev2_adam_dense_2_1_kernel_m_read_readvariableop0savev2_adam_dense_2_1_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop3savev2_adam_conv2d_4_1_kernel_v_read_readvariableop1savev2_adam_conv2d_4_1_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop2savev2_adam_dense_2_1_kernel_v_read_readvariableop0savev2_adam_dense_2_1_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
_output_shapes
: *
dtype0
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0Ã
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ¹
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
N*
T0
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*³
_input_shapes¡
: : : :  : : @:@:@@:@:
::	+:+: : : : : : : : : :  : : @:@:@@:@:
::	+:+: : :  : : @:@:@@:@:
::	+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:$ : : :  : : :, : : :( : :
 : :' : : :# : : : :	 :+ : : : :& : :+ '
%
_user_specified_namefile_prefix:" : : : : :* :% : : :! : : : : :) : : 
Î2

H__inference_sequential_1_layer_call_and_return_conditional_losses_361032

inputs+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2+
'conv2d_7_statefulpartitionedcall_args_1+
'conv2d_7_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tin
2*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669*-
_gradient_op_typePartitionedCall-360675*
Tout
2*-
config_proto

CPU

GPU2*0J 8µ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tin
2*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694*-
_gradient_op_typePartitionedCall-360700*
Tout
2*-
config_proto

CPU

GPU2*0J 8ß
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360719*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
Tin
2*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713´
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
Tout
2*M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736*
Tin
2*-
config_proto

CPU

GPU2*0J 8*-
_gradient_op_typePartitionedCall-360742µ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0'conv2d_7_statefulpartitionedcall_args_1'conv2d_7_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360767*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761*
Tin
2ß
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360786*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780*
Tin
2â
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-360840*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
Tin
2*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_360829Í
flatten_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tin
2*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859*-
_gradient_op_typePartitionedCall-360865*
Tout
2*-
config_proto

CPU

GPU2*0J 8£
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tin
2*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_360883*-
_gradient_op_typePartitionedCall-360889*
Tout
2*-
config_proto

CPU

GPU2*0J 8ÿ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_360920*-
_gradient_op_typePartitionedCall-360931ª
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-360961*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_360955*
Tin
2
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : :
 : : : : :	 : 
Ö	
Ü
C__inference_dense_3_layer_call_and_return_conditional_losses_360955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:	+*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ñ
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_360829

inputs
identityQ
dropout/rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0*
dtype0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ª
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
dropout/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
ïN
¬

!__inference__wrapped_model_360655
conv2d_4_input8
4sequential_1_conv2d_4_conv2d_readvariableop_resource9
5sequential_1_conv2d_4_biasadd_readvariableop_resource8
4sequential_1_conv2d_5_conv2d_readvariableop_resource9
5sequential_1_conv2d_5_biasadd_readvariableop_resource8
4sequential_1_conv2d_6_conv2d_readvariableop_resource9
5sequential_1_conv2d_6_biasadd_readvariableop_resource8
4sequential_1_conv2d_7_conv2d_readvariableop_resource9
5sequential_1_conv2d_7_biasadd_readvariableop_resource7
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identity¢,sequential_1/conv2d_4/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_4/Conv2D/ReadVariableOp¢,sequential_1/conv2d_5/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_5/Conv2D/ReadVariableOp¢,sequential_1/conv2d_6/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_6/Conv2D/ReadVariableOp¢,sequential_1/conv2d_7/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_7/Conv2D/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢*sequential_1/dense_3/MatMul/ReadVariableOpÖ
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
: *
dtype0Î
sequential_1/conv2d_4/Conv2DConv2Dconv2d_4_input3sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
T0Ì
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0¿
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  è
sequential_1/conv2d_5/Conv2DConv2D(sequential_1/conv2d_4/Relu:activations:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
T0Ì
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ¿
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
T0Æ
$sequential_1/max_pooling2d_2/MaxPoolMaxPool(sequential_1/conv2d_5/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
+sequential_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
: @*
dtype0í
sequential_1/conv2d_6/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:03sequential_1/conv2d_6/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
strides
*
T0*
paddingVALIDÌ
,sequential_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0¿
sequential_1/conv2d_6/BiasAddBiasAdd%sequential_1/conv2d_6/Conv2D:output:04sequential_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
sequential_1/conv2d_6/ReluRelu&sequential_1/conv2d_6/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
T0Ö
+sequential_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0è
sequential_1/conv2d_7/Conv2DConv2D(sequential_1/conv2d_6/Relu:activations:03sequential_1/conv2d_7/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
strides
*
T0*
paddingVALIDÌ
,sequential_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0¿
sequential_1/conv2d_7/BiasAddBiasAdd%sequential_1/conv2d_7/Conv2D:output:04sequential_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential_1/conv2d_7/ReluRelu&sequential_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_7/Relu:activations:0*
ksize
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

sequential_1/dropout_2/IdentityIdentity-sequential_1/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
$sequential_1/flatten_1/Reshape/shapeConst*
valueB"ÿÿÿÿ   *
_output_shapes
:*
dtype0µ
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_2/Identity:output:0-sequential_1/flatten_1/Reshape/shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0Î
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
µ
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:¶
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/dropout_3/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:	+*
dtype0µ
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+Ê
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:+*
dtype0µ
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp-^sequential_1/conv2d_6/BiasAdd/ReadVariableOp,^sequential_1/conv2d_6/Conv2D/ReadVariableOp-^sequential_1/conv2d_7/BiasAdd/ReadVariableOp,^sequential_1/conv2d_7/Conv2D/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2Z
+sequential_1/conv2d_7/Conv2D/ReadVariableOp+sequential_1/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2\
,sequential_1/conv2d_6/BiasAdd/ReadVariableOp,sequential_1/conv2d_6/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_6/Conv2D/ReadVariableOp+sequential_1/conv2d_6/Conv2D/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2\
,sequential_1/conv2d_7/BiasAdd/ReadVariableOp,sequential_1/conv2d_7/BiasAdd/ReadVariableOp: : :. *
(
_user_specified_nameconv2d_4_input: : : :
 : : : : :	 : 

c
E__inference_dropout_3_layer_call_and_return_conditional_losses_360927

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
©
ª
)__inference_conv2d_5_layer_call_fn_360705

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-360700*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
°
L
0__inference_max_pooling2d_2_layer_call_fn_360722

inputs
identityÇ
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-360719*
Tout
2*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
Tin
2
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs

c
E__inference_dropout_3_layer_call_and_return_conditional_losses_361381

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ñ
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_361312

inputs
identityQ
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ª
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0R
dropout/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
ü
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_360859

inputs
identity^
Reshape/shapeConst*
valueB"ÿÿÿÿ   *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
Ö	
Ü
C__inference_dense_3_layer_call_and_return_conditional_losses_361402

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	+i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:+v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
§@

H__inference_sequential_1_layer_call_and_return_conditional_losses_361258

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¼
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ¬
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
paddingVALID*
T0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: 
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:  *
dtype0Á
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
paddingVALID*
T0*
strides
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
max_pooling2d_2/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
*
ksize
¼
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @Æ
conv2d_6/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
strides
*
T0*
paddingVALID²
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
T0j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@¼
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0Á
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
strides
*
T0*
paddingVALID²
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0¬
max_pooling2d_3/MaxPoolMaxPoolconv2d_7/Relu:activations:0*
ksize
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
strides
*
paddingVALIDz
dropout_2/IdentityIdentity max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0h
flatten_1/Reshape/shapeConst*
valueB"ÿÿÿÿ   *
dtype0*
_output_shapes
:
flatten_1/ReshapeReshapedropout_2/Identity:output:0 flatten_1/Reshape/shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0´
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0±
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0a
dense_2/ReluReludense_2/BiasAdd:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0m
dropout_3/IdentityIdentitydense_2/Relu:activations:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0³
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:	+*
dtype0
dense_3/MatMulMatMuldropout_3/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0°
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:+*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0ï
IdentityIdentitydense_3/Softmax:softmax:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*
T0"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : :
 : : : : :	 : 

¡
-__inference_sequential_1_layer_call_fn_361048
conv2d_4_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*-
config_proto

CPU

GPU2*0J 8*
Tin
2*-
_gradient_op_typePartitionedCall-361033*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_361032*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 : : : :. *
(
_user_specified_nameconv2d_4_input: : : :
 

g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780

inputs
identity¢
MaxPoolMaxPoolinputs*
ksize
*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
strides
*
paddingVALID{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs

Ý
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpª
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
strides
*
T0*
paddingVALID 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0¥
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
Q
conv2d_4_input?
 serving_default_conv2d_4_input:0ÿÿÿÿÿÿÿÿÿ  ;
dense_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ+tensorflow/serving/predict:ûÏ
H
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
¬_default_save_signature
+­&call_and_return_all_conditional_losses
®__call__"D
_tf_keras_sequentialõC{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Á
regularization_losses
	variables
trainable_variables
	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"°
_tf_keras_layer{"class_name": "InputLayer", "name": "conv2d_4_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 32, 32, 1], "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "name": "conv2d_4_input"}}
£

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+±&call_and_return_all_conditional_losses
²__call__"ü
_tf_keras_layerâ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 32, 32, 1], "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
ï

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+³&call_and_return_all_conditional_losses
´__call__"È
_tf_keras_layer®{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ÿ
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"î
_tf_keras_layerÔ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ï

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"È
_tf_keras_layer®{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ï

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"È
_tf_keras_layer®{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ÿ
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"î
_tf_keras_layerÔ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
±
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+½&call_and_return_all_conditional_losses
¾__call__" 
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
²
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"¡
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ö

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}
±
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__" 
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
÷

Ikernel
Jbias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Ã
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemmmm'm(m-m.m?m@mImJmv v¡v¢v£'v¤(v¥-v¦.v§?v¨@v©IvªJv«"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11"
trackable_list_wrapper
v
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11"
trackable_list_wrapper
»
Tnon_trainable_variables

Ulayers
regularization_losses
	variables
Vmetrics
trainable_variables
Wlayer_regularization_losses
®__call__
¬_default_save_signature
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
-
Çserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Xnon_trainable_variables

Ylayers
regularization_losses
	variables
Zmetrics
trainable_variables
[layer_regularization_losses
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv2d_4_1/kernel
: 2conv2d_4_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

\non_trainable_variables

]layers
regularization_losses
	variables
^metrics
trainable_variables
_layer_regularization_losses
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_5/kernel
: 2conv2d_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

`non_trainable_variables

alayers
regularization_losses
 	variables
bmetrics
!trainable_variables
clayer_regularization_losses
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

dnon_trainable_variables

elayers
#regularization_losses
$	variables
fmetrics
%trainable_variables
glayer_regularization_losses
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_6/kernel
:@2conv2d_6/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper

hnon_trainable_variables

ilayers
)regularization_losses
*	variables
jmetrics
+trainable_variables
klayer_regularization_losses
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper

lnon_trainable_variables

mlayers
/regularization_losses
0	variables
nmetrics
1trainable_variables
olayer_regularization_losses
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

pnon_trainable_variables

qlayers
3regularization_losses
4	variables
rmetrics
5trainable_variables
slayer_regularization_losses
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

tnon_trainable_variables

ulayers
7regularization_losses
8	variables
vmetrics
9trainable_variables
wlayer_regularization_losses
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

xnon_trainable_variables

ylayers
;regularization_losses
<	variables
zmetrics
=trainable_variables
{layer_regularization_losses
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_2_1/kernel
:2dense_2_1/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper

|non_trainable_variables

}layers
Aregularization_losses
B	variables
~metrics
Ctrainable_variables
layer_regularization_losses
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
non_trainable_variables
layers
Eregularization_losses
F	variables
metrics
Gtrainable_variables
 layer_regularization_losses
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
!:	+2dense_3/kernel
:+2dense_3/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
¡
non_trainable_variables
layers
Kregularization_losses
L	variables
metrics
Mtrainable_variables
 layer_regularization_losses
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
	7

8
9
10"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
£

total

count

_fn_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
+È&call_and_return_all_conditional_losses
É__call__"å
_tf_keras_layerË{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
non_trainable_variables
layers
regularization_losses
	variables
metrics
trainable_variables
 layer_regularization_losses
É__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:. 2Adam/conv2d_4_1/kernel/m
":  2Adam/conv2d_4_1/bias/m
.:,  2Adam/conv2d_5/kernel/m
 : 2Adam/conv2d_5/bias/m
.:, @2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
.:,@@2Adam/conv2d_7/kernel/m
 :@2Adam/conv2d_7/bias/m
):'
2Adam/dense_2_1/kernel/m
": 2Adam/dense_2_1/bias/m
&:$	+2Adam/dense_3/kernel/m
:+2Adam/dense_3/bias/m
0:. 2Adam/conv2d_4_1/kernel/v
":  2Adam/conv2d_4_1/bias/v
.:,  2Adam/conv2d_5/kernel/v
 : 2Adam/conv2d_5/bias/v
.:, @2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
.:,@@2Adam/conv2d_7/kernel/v
 :@2Adam/conv2d_7/bias/v
):'
2Adam/dense_2_1/kernel/v
": 2Adam/dense_2_1/bias/v
&:$	+2Adam/dense_3/kernel/v
:+2Adam/dense_3/bias/v
î2ë
!__inference__wrapped_model_360655Å
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *5¢2
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
î2ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_361002
H__inference_sequential_1_layer_call_and_return_conditional_losses_360973
H__inference_sequential_1_layer_call_and_return_conditional_losses_361258
H__inference_sequential_1_layer_call_and_return_conditional_losses_361206À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_sequential_1_layer_call_fn_361292
-__inference_sequential_1_layer_call_fn_361048
-__inference_sequential_1_layer_call_fn_361275
-__inference_sequential_1_layer_call_fn_361095À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
£2 
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
)__inference_conv2d_4_layer_call_fn_360680×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
£2 
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
2
)__inference_conv2d_5_layer_call_fn_360705×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
³2°
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_2_layer_call_fn_360722à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
£2 
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
2
)__inference_conv2d_6_layer_call_fn_360747×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
£2 
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
)__inference_conv2d_7_layer_call_fn_360772×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
³2°
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_3_layer_call_fn_360789à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
È2Å
E__inference_dropout_2_layer_call_and_return_conditional_losses_361312
E__inference_dropout_2_layer_call_and_return_conditional_losses_361317´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_2_layer_call_fn_361322
*__inference_dropout_2_layer_call_fn_361327´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_361333¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_1_layer_call_fn_361338¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_361349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_361356¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
E__inference_dropout_3_layer_call_and_return_conditional_losses_361376
E__inference_dropout_3_layer_call_and_return_conditional_losses_361381´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_3_layer_call_fn_361386
*__inference_dropout_3_layer_call_fn_361391´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_361402¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_361409¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:B8
$__inference_signature_wrapper_361122conv2d_4_input
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 Ê
H__inference_sequential_1_layer_call_and_return_conditional_losses_360973~'(-.?@IJG¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 ¥
C__inference_dense_2_layer_call_and_return_conditional_losses_361349^?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_3_layer_call_and_return_conditional_losses_361381^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
!__inference__wrapped_model_360655'(-.?@IJ?¢<
5¢2
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
ª "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ+±
)__inference_conv2d_7_layer_call_fn_360772-.I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
-__inference_sequential_1_layer_call_fn_361292i'(-.?@IJ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ+Æ
0__inference_max_pooling2d_2_layer_call_fn_360722R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_flatten_1_layer_call_fn_361338T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_360780R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
E__inference_flatten_1_layer_call_and_return_conditional_losses_361333a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ù
D__inference_conv2d_4_layer_call_and_return_conditional_losses_360669I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Â
H__inference_sequential_1_layer_call_and_return_conditional_losses_361206v'(-.?@IJ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 §
E__inference_dropout_3_layer_call_and_return_conditional_losses_361376^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 î
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_360713R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ù
D__inference_conv2d_6_layer_call_and_return_conditional_losses_360736'(I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 |
(__inference_dense_3_layer_call_fn_361409PIJ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ+±
)__inference_conv2d_6_layer_call_fn_360747'(I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
*__inference_dropout_2_layer_call_fn_361327_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@½
$__inference_signature_wrapper_361122'(-.?@IJQ¢N
¢ 
GªD
B
conv2d_4_input0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ+¢
-__inference_sequential_1_layer_call_fn_361048q'(-.?@IJG¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ+±
)__inference_conv2d_5_layer_call_fn_360705I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ù
D__inference_conv2d_5_layer_call_and_return_conditional_losses_360694I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
-__inference_sequential_1_layer_call_fn_361275i'(-.?@IJ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ+Ù
D__inference_conv2d_7_layer_call_and_return_conditional_losses_360761-.I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 µ
E__inference_dropout_2_layer_call_and_return_conditional_losses_361312l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_dropout_3_layer_call_fn_361391Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÆ
0__inference_max_pooling2d_3_layer_call_fn_360789R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
E__inference_dropout_2_layer_call_and_return_conditional_losses_361317l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ±
)__inference_conv2d_4_layer_call_fn_360680I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ê
H__inference_sequential_1_layer_call_and_return_conditional_losses_361002~'(-.?@IJG¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 
*__inference_dropout_2_layer_call_fn_361322_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¢
-__inference_sequential_1_layer_call_fn_361095q'(-.?@IJG¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ+
*__inference_dropout_3_layer_call_fn_361386Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÂ
H__inference_sequential_1_layer_call_and_return_conditional_losses_361258v'(-.?@IJ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 ¤
C__inference_dense_3_layer_call_and_return_conditional_losses_361402]IJ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 }
(__inference_dense_2_layer_call_fn_361356Q?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
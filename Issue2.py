import numpy as np 
import paddle.fluid as fluid 
import math 
from paddle.fluid.layers import relu, prelu, leaky_relu
s = 64 
m1 = 1.35 
m2 = 0.5 
m3 = 0.35

place = fluid.CPUPlace()
sess = fluid.Executor( place )

class BatchNormConv2D( object ):
    def __init__( self, filters = 8, \
                  kernel_w = 3, kernel_h= 3, 
                  stride_h = 1, stride_v = 1,
                  pad_h = 0, pad_v = 0,
                  dilation_h = 1, dilation_v = 1,
                  activation = None,
                  use_bias = True,
                  kernel_init = fluid.initializer.MSRAInitializer(), bias_init = fluid.initializer.ConstantInitializer(),
                  groups = 1,
                  kernel_regu = None, bias_regu = None,
                  activation_regu = None,
                  bias_constraint = None,
                  reuse = False,
                  trainable = True,
                  name = "BNConv2D" ):
        self.filters = filters
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.stride_h = stride_h
        self.stride_v = stride_v
        self.pad_h = pad_h
        self.pad_v = pad_v
        self.dilation_v = dilation_v
        self.dilation_h = dilation_h
        self.activation = activation 
        self.use_bias = use_bias,
        self.kernel_init = kernel_init 
        self.bias_init = bias_init 
        self.groups = groups
        self.kernel_regu = kernel_regu
        self.bias_regu = bias_regu 
        self.activation_regu = activation_regu
        self.bias_constraint = bias_constraint
        self.reuse = reuse,
        self.trainable = trainable
        self.name = name
    
    def __call__( self, inputs, transform_input = False ):
        self.w = self.name + "_w"
        self.b = self.name + "_b"
        w_attr = fluid.ParamAttr( name = self.w, 
                                  initializer = self.kernel_init,
                                  regularizer = self.kernel_regu,
                                  trainable = self.trainable )
        b_attr = fluid.ParamAttr( name = self.b, 
                                  initializer = self.bias_init,
                                  regularizer = self.bias_regu,
                                  trainable = self.trainable )
        if transform_input:
            inputs = fluid.layers.transpose( inputs, perm = [ 0, 3, 1, 2 ] )
        print( self.name, "input shape:", inputs.shape )
        """
        [ n, c, h, w ] = inputs.shape
        # calc padding for each side
        filter_w = self.dilation_h * ( self.kernel_w - 1 ) + 1 
        filter_h = self.dilation_v * ( self.kernel_h - 1 ) + 1
        if self.padding.lower() == 'valid':
            width = math.ceil( (  w - filter_w ) / self.stride_h ) + 1
            height = math.ceil( ( h - filter_h ) / self.stride_v ) + 1
        elif self.padding.lower() == "same":
            width = math.ceil( w / self.stride_h )
            height = math.ceil( h / self.stride_v )
        else:
            NotImplementedError( "Not an implemented padding for Conv2D: %s", self.padding )
        print( "width", width, "height", height, [ n, c, h, w ] )
        restructured_w = ( width - 1 ) * self.stride_h + filter_w
        restructured_h = ( height - 1) * self.stride_v + filter_h 

        pad_h = max( math.ceil( ( restructured_w - w ) / 2 ), 0 )
        pad_v = max( math.ceil( ( restructured_h - h ) / 2 ), 0 )

        self.pad_h, self.pad_v = pad_h, pad_v
        print( "pad_v", pad_v, "pad_h", pad_h )
        """
        self.layer_out = fluid.layers.conv2d( input = inputs,
                                              num_filters = self.filters,
                                              filter_size = ( self.kernel_h, self.kernel_w ),
                                              stride = ( self.stride_v, self.stride_h ),
                                              padding = ( self.pad_v, self.pad_h ),
                                              dilation = ( self.dilation_v, self.dilation_h ),
                                              act = None, 
                                              groups = self.groups,
                                              param_attr = w_attr,
                                              bias_attr = b_attr,
                                              name = self.name )
        self.layer_out = fluid.layers.batch_norm( self.layer_out, act = None, use_global_stats=True )
        if self.activation is not None:
            self.layer_out = self.activation( self.layer_out )
        print( self.name, "output shape", self.layer_out.shape )
        print( "--------------------------" )
        return self.layer_out

    def __repr__( self ):

        return self.name  

    def get_weights( self, sess = None ):

        assert ( self.w is not None ) and ( self. b is not None ), "weights in the dense layer should be initialized"

        # place = fluid.CPUPlace()
        # exe = fluid.Executor(place)
        # exe.run(fluid.default_startup_program())
        w = fluid.global_scope().find_var( self.w ).get_tensor()
        b = fluid.global_scope().find_var( self.b ).get_tensor()

        return np.array( w ), np.array( b )

class Dense( object ):

    def __init__( self, units, 
                  activation = None, use_bias = True, 
                  kernel_init = fluid.initializer.MSRAInitializer(), bias_init = fluid.initializer.ConstantInitializer(),
                  kernel_regu = None, bias_regu = None,
                  activation_regu = None,
                  bias_constraint = None,
                  reuse = False,
                  trainable = True,
                  name = "Dense" ):

        self.units = units
        self.activation  = activation
        self.use_bias = use_bias 
        self.kernel_init = kernel_init 
        self.bias_init = bias_init 
        self.kernel_regu = kernel_regu
        self.bias_regu = bias_regu 
        self.activation_regu = activation_regu
        self.bias_constraint = bias_constraint
        self.reuse = reuse
        self.trainable = trainable
        self.name = name
    
    def __call__( self, inputs ):

        self.w = self.name + "_w"
        self.b = self.name + "_b"
        w_attr = fluid.ParamAttr( name = self.w, 
                                  initializer = self.kernel_init,
                                  regularizer = self.kernel_regu,
                                  trainable = self.trainable )
        b_attr = fluid.ParamAttr( name = self.b, 
                                  initializer = self.bias_init,
                                  regularizer = self.bias_regu,
                                  trainable = self.trainable )

        self.layer_out = fluid.layers.fc( inputs,
                                          self.units,
                                          num_flatten_dims = 1,
                                          act = None, 
                                          param_attr = w_attr,
                                          bias_attr = b_attr,
                                          name = self.name )
        if self.activation is not None:
            self.layer_out = self.activation( self.layer_out )
        return self.layer_out

    def __repr__( self ):

        return self.name  

    def get_weights( self, sess = None ):

        assert ( self.w is not None ) and ( self. b is not None ), "weights in the dense layer should be initialized"

        # place = fluid.CPUPlace()
        # exe = fluid.Executor(place)
        # exe.run(fluid.default_startup_program())
        w = fluid.global_scope().find_var( self.w ).get_tensor()
        b = fluid.global_scope().find_var( self.b ).get_tensor()

        return np.array( w ), np.array( b )

class ArcLinear( object ):

    def __init__( self, in_size, out_size, m = 4, phiflag = True ):
        self.in_size = in_size
        self.out_size = out_size
        matrix = np.random.uniform( -1.0, 1.0, [ in_size, out_size ] )
        norm = np.linalg.norm( matrix, 2, 1, True )
        matrix = matrix / norm
        matrix = matrix.astype( np.float32 )
        param_attr = fluid.ParamAttr( trainable = False, name = "eye",
                                      initializer = fluid.initializer.NumpyArrayInitializer( matrix ) )
        self.weight = fluid.layers.create_parameter( [ in_size, out_size ], 
                                                     "float32", 
                                                     name = "weight",
                                                     attr = param_attr,
                                                     is_bias = False )
    
    def __call__( self, inputs ):
        x = inputs 
        ww = fluid.layers.l2_normalize( self.weight, axis = 1, name = "weight norm" ) 
        cos_theta = fluid.layers.mul( x, ww )
        return cos_theta * s

class ArcLoss( object ):
    def __init__( self, gamma=0., class_num = 10575 ):
        self.class_size = class_num

    def __call__( self, inputs, target ):
        cos_theta = inputs
        cos_theta = cos_theta / s 
        truth_cos_theta = fluid.layers.gather( cos_theta, target )
        truth_theta = fluid.layers.acos( truth_cos_theta )

        truth_theta = m1 * truth_theta + m2 
        truth_theta = fluid.layers.cos( truth_theta )
        truth_theta -= m3 
        diff = truth_theta - truth_cos_theta
        diff = fluid.layers.reshape( diff, [ -1, 1 ], inplace = True )
        diff = fluid.layers.expand( diff, [ 1, self.class_size ] )
        
        index = fluid.layers.one_hot( target, self.class_size )
        index = fluid.layers.cast( index, "float32"  )
        diff = fluid.layers.cast( diff, "float32" )
        # index = fluid.layers.reshape( index, [ -1, self.class_size ], inplace = True )  
        print( "inputs shape", inputs.shape )
        print( "index shape", index.shape )
        print( "diff shape", diff.shape )
        print( "cos_theta shape", cos_theta.shape )
        index = index * diff
        cos_theta += index

        cos_theta *= s 

        return cos_theta

sample_pos = fluid.layers.data( name = "sample_pos", shape = [ 125, 125, 3 ], 
                                            dtype = "float32", append_batch_size = True,
                                            stop_gradient = False ) 
label_pos = fluid.layers.data( name = "label_pos", shape = [ 1 ], dtype = "int32", append_batch_size = True ) 
lr = fluid.layers.data( name = "lr", shape = [ 1 ], dtype = "float32", append_batch_size = False )
conv_1_1_obj = BatchNormConv2D( 64, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 2, stride_v = 2,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_1_1" )
conv_1_2_obj = BatchNormConv2D( 64, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_1_2" )
conv_1_3_obj = BatchNormConv2D( 64, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_1_3" )

conv_2_1_obj = BatchNormConv2D( 128, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 2, stride_v = 2,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_2_1" )
conv_2_2_obj = BatchNormConv2D( 128, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_2_2" )
conv_2_3_obj = BatchNormConv2D( 128, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_2_3" )

conv_2_4_obj = BatchNormConv2D( 128, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_2_4" )
conv_2_5_obj = BatchNormConv2D( 128, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_2_5" )

conv_3_1_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 2, stride_v = 2,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_1" )
conv_3_2_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_2" )
conv_3_3_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_3" )

conv_3_4_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_4" )
conv_3_5_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_5" )

conv_3_6_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_6" )
conv_3_7_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_7" )

conv_3_8_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_8" )
conv_3_9_obj = BatchNormConv2D( 256, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_3_9" )

conv_4_1_obj = BatchNormConv2D( 512, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 2, stride_v = 2,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_4_1" )
conv_4_2_obj = BatchNormConv2D( 512, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_4_2" )
conv_4_3_obj = BatchNormConv2D( 512, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 1, stride_v = 1,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_4_3" )

conv_4_4_obj = BatchNormConv2D( 512, 
                            kernel_w = 3, kernel_h = 3, 
                            stride_h = 2, stride_v = 2,
                            pad_h = 1, pad_v = 1,
                            activation = leaky_relu,
                            name = "conv_4_4" )

pre_embed_obj = Dense( 512, activation = leaky_relu, name = "pre_embed" )
embed_obj = Dense( 512, activation = leaky_relu, name = "embed" )
loss_obj = Dense( 10575, activation = leaky_relu, name = "logit" )
another_obj = Dense( 200, activation = leaky_relu, name = "another" )

out = conv_1_1_obj( sample_pos, True )
out = out + conv_1_3_obj( conv_1_2_obj( out ) )

out = conv_2_1_obj( out )
out = out + conv_2_3_obj( conv_2_2_obj( out ) )
out = out + conv_2_5_obj( conv_2_4_obj( out ) )

out = conv_3_1_obj( out )
out = out + conv_3_3_obj( conv_3_2_obj( out ) )
out = out + conv_3_5_obj( conv_3_4_obj( out ) )
out = out + conv_3_7_obj( conv_3_6_obj( out ) )
out = out + conv_3_9_obj( conv_3_8_obj( out ) )

out = conv_4_1_obj( out )
out_tmp = conv_4_3_obj( conv_4_2_obj( out ) )
print( "HAHAHA", out.shape, out_tmp.shape )
out = out + out_tmp


flatten = fluid.layers.flatten( out, axis = 1 )
flatten *= fluid.layers.gaussian_random( flatten.shape )
another = another_obj( flatten )
embed = pre_embed_obj( flatten )
logit = embed_obj( embed )
logit = loss_obj( logit )
loss = fluid.layers.softmax_with_cross_entropy( logit, fluid.layers.cast( label_pos, "int64" ), soft_label = False)
loss = fluid.layers.mean( loss ) + fluid.layers.mean( another )


startup_program = fluid.default_startup_program()
main_program = fluid.default_main_program()
test_program = main_program.clone( for_test = True )
optim = fluid.optimizer.AdamOptimizer( lr, beta1 = 0.9, beta2 = 0.99 )
_, grad_list = optim.minimize( loss )
sess.run( fluid.default_startup_program() )


bs = np.random.normal( size = (8, 125, 125, 3) ).astype( np.float32 )
bl = np.random.randint( 10575, size = ( 8, 1 ) ).astype( np.int32 )
learningrate = 0.001

print( bs.shape, bl.shape )
[ loss ] = sess.run( program = main_program,
                     feed = { sample_pos.name : bs, 
                                label_pos.name: bl,
                                lr.name : learningrate },
                     fetch_list = [ loss.name ] )
print( loss )


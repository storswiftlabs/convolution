use dep::std;
mod quantize;
mod tensor;
use tensor::Tensor;


struct Conv2D<W_LEN, B_LEN> {
    // input: Tensor<IN_LEN, 3>, 
    weight: Tensor<W_LEN, 4>,
    bias: Tensor<B_LEN, 2>,  
    stride: comptime Field,
    padding: comptime Field,
}

impl<W_LEN, B_LEN> Conv2D<W_LEN, B_LEN> {
    fn new(weight: Tensor<W_LEN, 4>, bias: Tensor<B_LEN, 2>, stride: comptime Field, padding: comptime Field) -> Self {
        Self { weight, bias, stride, padding }
    }

    fn pad<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>) -> Tensor<OUT_LEN, 3> {
        let mut outputs = Tensor::new_quantized([inputs.zero_point; OUT_LEN], [inputs.shape[0], inputs.shape[1]+2*self.padding, inputs.shape[2]+2*self.padding], inputs.zero_point, inputs.scale);
        for c in 0..inputs.shape[0] {
            for h in 0..inputs.shape[1]{
                for w in 0..inputs.shape[2]{
                    outputs.values = outputs.set([c, h+self.padding, w+self.padding], inputs.get([c, h, w]));
                }
            }
        }
        outputs
    }

    fn forward<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>) -> Tensor<OUT_LEN, 3> {
        let pad_outputs = self.pad(inputs);
        let mut outputs = Tensor::new_quantized([pad_outputs.zero_point; OUT_LEN], [B_LEN, (pad_outputs.shape[1] - self.weight.shape[1])/self.stride + 1, (pad_outputs.shape[2] - self.weight.shape[2])/self.stride + 1], pad_outputs.zero_point, pad_outputs.scale);
        for out_c in 0..outputs.shape[0] {
            for h_step in 0..outputs.shape[1] {
                for w_step in 0..outputs.shape[2] {
                    let mut w_step_index: comptime Field = w_step*self.stride;
                    let mut h_step_index: comptime Field = h_step*self.stride;
                    // let mut sum = outputs.zero_point;
                    let mut out_slice = outputs.slice([0, h_step_index, w_step_index], [outputs.shape[0], h_step_index+self.weight.shape[1], w_step_index+self.weight.shape[2]]);
                    // for c in 0..self.weight.shape[0] {
                    //     for h in 0..self.weight.shape[1] {
                    //         for w in 0..self.weight.shape[2]{
                    //             // sum = quantize::add(sum, quantize::mul(self.weight.get([c, h, w]), , pad_outputs.get([c, h_step_index+h, w_step_index+w])));
                    //         }
                    //     }
                    // }
                    let mut weight_slice = self.weight.slice([out_c,0, 0, 0], [out_c+1,self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]]);
                    let product = out_slice.hadamard_product(weight_slice);
                    let sum = product.sum();
                    outputs.values = outputs.set([out_c, h_step, w_step], sum);
                }
            }
        }
        outputs
    }
}

struct Pool2D {
    stride: comptime Field,
    padding: comptime Field,
    kernel_size: comptime Field,
}

impl Pool2D {
    fn pad<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>) -> Tensor<OUT_LEN, 3> {
        let mut outputs = Tensor::new_quantized([inputs.zero_point; OUT_LEN], [inputs.shape[0], inputs.shape[1]+2*self.padding, inputs.shape[2]+2*self.padding], inputs.zero_point, inputs.scale);
        for c in 0..inputs.shape[0] {
            for h in 0..inputs.shape[1]{
                for w in 0..inputs.shape[2]{
                    outputs.values = outputs.set([c, h+self.padding, w+self.padding], inputs.get([c, h, w]));
                }
            }
        }
        outputs
    }

    fn pooling<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>, pooling_type: u8) -> Tensor<OUT_LEN, 3> {
        let pad_outputs = self.pad(inputs);
        let mut outputs = Tensor::new_quantized([pad_outputs.zero_point; OUT_LEN], [pad_outputs.shape[0], (pad_outputs.shape[1] - self.kernel_size)/self.stride + 1, (pad_outputs.shape[2] - self.kernel_size)/self.stride + 1], pad_outputs.zero_point, pad_outputs.scale);
        for out_c in 0..outputs.shape[0] {
            for h_step in 0..outputs.shape[1] {
                for w_step in 0..outputs.shape[2] {
                    let mut w_step_index = w_step*self.stride;
                    let mut h_step_index = h_step*self.stride;
                    // let mut sum = outputs.zero_point;
                    let mut slice = outputs.slice([out_c, h_step_index, w_step_index], [out_c+1, h_step_index+self.kernel_size, w_step_index+self.kernel_size]);
                    let mut v = 0;
                    if pooling_type==0 {
                        v = slice.max();
                    }
                    else if pooling_type==1 {
                        v = slice.avg();
                    }
                    outputs.values = outputs.set([out_c, h_step, w_step], v);
                }
            }
        }
        outputs
    }

    fn max_pooling<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>) -> Tensor<OUT_LEN, 3> {
        self.pooling(inputs, 0)
    }

    fn avg_pooling<IN_LEN, OUT_LEN>(self, inputs:Tensor<IN_LEN, 3>) -> Tensor<OUT_LEN, 3> {
        self.pooling(inputs, 1)
    }
}

struct Linear<W_LEN, B_LEN> {
    weight: Tensor<W_LEN, 2>,
    bias: Tensor<B_LEN, 2>,  
}

impl<W_LEN, B_LEN> Linear<W_LEN, B_LEN> {
    fn new(weight: Tensor<W_LEN, 2>, bias: Tensor<B_LEN, 2>) -> Self {
        Self { weight, bias}
    }

    fn forward<IN_LEN>(self, inputs:Tensor<IN_LEN, 2>) -> Tensor<B_LEN, 2> {
        // let outputs = Tensor::new_quantized([inputs.zero_point; OUT_LEN], [1, B_LEN], inputs.zero_point, inputs.scale);
        let mut outputs = self.weight.mul(inputs);
        outputs = outputs.add(self.bias);
        outputs
    }
}

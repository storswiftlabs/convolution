use dep::std::println;

struct Tensor<N, S> {
    values: [Field; N],                     // Array to hold tensor values
    shape: [comptime Field; S],             // Shape of the tensor
    zero_point: Field,                      // Zero point for quantization
    scale: Field,                // Reciprocal of the scaling factor for quantization
}

impl<N, S> Tensor<N, S> {
    // Constructor for a Tensor with default quantization parameters
    fn new(values: [Field; N], shape: [comptime Field; S]) -> Self {
        Self { values, shape, zero_point: 0, scale: 1 }
    }

    // Constructor for a quantized Tensor with custom quantization parameters
    fn new_quantized(values: [Field; N], shape: [comptime Field; S], zero_point: Field, scale: Field) -> Self {
        Self { values, shape, zero_point, scale }
    }

    fn get_index(self, index_arr: [comptime Field; S]) -> Field {
        let mut index = 0;
        for i in 0..self.shape.len() {
            let mut product = index_arr[i];
            for j in (i+1)..self.shape.len() {
                product*=self.shape[j];
            }
            index+=product;
        }
        index
    }

    fn get(self, index_arr: [comptime Field; S]) -> Field {
        self.values[self.get_index(index_arr)]
    }

    fn set(mut self, index_arr: [comptime Field; S], value: Field) -> [Field; N] {
        self.values[self.get_index(index_arr)] = value;
        self.values
    }

    fn reshape<M>(self, shape: [comptime Field; M]) -> Tensor<N, M> {
        let mut result =  Tensor {values: [0; N], shape: shape, zero_point: self.zero_point, scale: self.scale};
        result
    }

    fn slice<M>(self, start: [comptime Field; S], end: [comptime Field; S]) -> Tensor<M, S> {
        let mut values = [0; M];
        let mut shape = [0; S];
        let mut result_len = 1;
        for i in 0..self.shape.len() {
            assert(start[i] as u32<end[i] as u32);
            assert(start[i] as u32<self.shape[i] as u32);
            assert(end[i] as u32<=self.shape[i] as u32);
            shape[i] = end[i] - start[i];
            result_len*=(end[i] - start[i]);
        }
        let mut values_slice = slice(self.values, self.shape, start, end, 0);
        for i in 0..result_len{
            values[i] = values_slice[i];
        }
        let mut result =  Tensor {values: values, shape: shape, zero_point: self.zero_point, scale: self.scale};
        result
    }

    fn print(self) {
        dep::std::println(self.values);
    }
}


fn slice<N, M, P>(tensor: [Field; N], shape: [comptime Field; M], start: [comptime Field; M], end: [comptime Field; M], dim: comptime Field) -> [Field; N] {
    let mut result = [0; N];
    if (shape.len() == dim+1) {
        for i in 0..(end[dim]-start[dim]) {
            result[i] = tensor[start[0]+i];
        }
        result
    }
    else {
        let mut d = 1;
        for j in (dim+1)..shape.len(){
            d *= shape[j];
        }
        let mut sub_result_len: comptime Field = end[dim+1]-start[dim+1];
        for n in (dim+2)..shape.len() {
            sub_result_len *= end[n]-start[n];
        }
        for i in start[dim]..end[dim]{
            let mut sub_tensor = [0; N];
            let mut index = i*d;
            for n in 0..d {
                sub_tensor[n] = tensor[index+n];
            }
            let sub_result = slice(sub_tensor, shape, start, end, dim+1);
            for n in 0..sub_result_len {
                result[(i-start[dim])*sub_result_len+n] = sub_result[n];
            }
        }
        result
    }
}

fn main() {
    // let tensor1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
    // let result = slice(tensor1, [18], [14], [18], 0);
    // println(result);
    let tensor1 = Tensor::new_quantized([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [3, 3, 2], 0, 1);   
    let mut result = tensor1.slice([2, 1, 0], [3, 3, 2]);  
    println(result);   
    assert(result.values==[15, 16, 17, 18]);
}

#[test]
fn test_slice() {
    let tensor1 = Tensor::new_quantized([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 0, 1);   
    let mut result = tensor1.slice([1,1], [3, 3]);  
    println(result);   
    assert(result.values==[5, 6, 8, 9]);
}


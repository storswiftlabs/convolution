use dep::std::println;
mod quantize;

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
        let mut result =  Tensor {values: [1; M], shape: start, zero_point: self.zero_point, scale: self.scale};
        for i in 0..self.shape.len() {
            assert(start[i] as u32<end[i] as u32);
            assert(start[i] as u32<self.shape[i] as u32);
            assert(end[i] as u32<=self.shape[i] as u32);
            result.shape[i] = end[i] - start[i];
        }
        let mut result_index = [0; S];
        let mut index = start;
        for k in 0..start.len() {
            for m in (k+1)..start.len() {
                for j in 0..(end[k] - start[k]) {
                    for i in 0..(end[m] - start[m]) {
                        result_index[k] = j;
                        index[k] = start[k]+j;
                        result_index[m] = i;
                        index[m] = start[m]+i;
                        result.values = result.set(result_index, self.get(index));
                        println(result_index);   
                        println(index);   
                        println(self.get(index));  
                    } 
                }
            }
        }
        println(result);   
        result
    }

    fn mul<M, P>(self: Tensor<N, 2>, tensor: Tensor<P, 2>) -> Tensor<M, 2>{
        assert(self.shape.len()==2);
        let mut result =  Tensor {values: [0; M], shape: [self.shape[0], tensor.shape[1]], zero_point: self.zero_point, scale: self.scale};
        if (self.zero_point==0) & (self.scale==1) & (tensor.zero_point==0) & (tensor.scale==1) {
            result.values = mul(self.values, tensor.values, self.shape, tensor.shape);
            result
        }
        else {
            assert(self.shape[1]==tensor.shape[0]);
            for i in 0..self.shape[0]{
                for j in 0..tensor.shape[1]{
                    let mut element = result.zero_point;
                    for k in 0..self.shape[1]{
                        element = quantize::add(quantize_mul(self.values[i * self.shape[1] + k], self.zero_point, self.scale, tensor.values[k * tensor.shape[1] + j], tensor.zero_point, tensor.scale, result.zero_point, result.scale), result.zero_point, result.scale, element, result.zero_point, result.scale, result.zero_point, result.scale);
                    }
                    result.values[i*tensor.shape[1]+j] = element;
                }
            }
            result
        }
    }


    fn print(self) {
        dep::std::println(self.values);
    }
}

fn quantize_mul(x: Field, x_z: Field, x_s: Field, y: Field, y_z: Field, y_s: Field, z_z: Field, z_s: Field) -> Field {
    let mut z: Field = (((quantize::mul(x, x_z, x_s, y, y_z, y_s, z_z, z_s) * x_s * y_s) as u126/(x_s * y_s) as u126) as Field);
    z
} 

fn mul<M, N, P>(tensor1: [Field; M], tensor2: [Field; P], tensor1_shape: [comptime Field; 2], tensor2_shape: [comptime Field; 2]) -> [Field; N]{
    // Ensure that the number of columns in the first tensor matches the number of rows in the second tensor
    assert(tensor1_shape[1] == tensor2_shape[0]);
    assert(N == tensor1_shape[0]*tensor2_shape[1]);
    let mut result = [0; N];
    
    // Iterate over rows in the first tensor
    for i in 0..tensor1_shape[0] {
        // Iterate over columns in the second tensor
        for j in 0..tensor2_shape[1] {
            let mut element = 0;
            
            // Perform dot product of row from first tensor and column from second tensor
            for k in 0..tensor1_shape[1] {
                element += tensor1[i * tensor1_shape[1] + k] * tensor2[k * tensor2_shape[1] + j];
            }
            
            // Store the resulting element in the appropriate position in the result tensor
            result[i * tensor2_shape[1] + j] = element;
        }
    }
    
    // Return the resulting tensor
    result
}

fn main() {
    let tensor1 = Tensor::new_quantized([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3], 0, 1);   
    let mut result = tensor1.slice([0,0], [2, 2]);  
    println(result);   
    assert(result.values==[1, 1, 1, 1]);
    // assert(result.values==[0, 0, 0, 0]);
}

#[test]
fn test_slice() {
    let tensor1 = Tensor::new_quantized([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 0, 1);   
    let mut result = tensor1.slice([1,1], [3, 3]);  
    println(result);   
    assert(result.values==[5, 6, 8, 9]);
}

#[test]
fn test_mul1() {
    let tensor1 = Tensor::new_quantized([999900, 1000200, 1000300, 1000400, 1000500, 999400], [2, 3], 1000000, 100);    // -quantize-> [-1, 2, 3, 4, 5, -6]
    let tensor2 = Tensor::new_quantized([1000700, 1000800, 1000900, 1001000, 1001100, 998800], [3, 2], 1000000, 100);    // -quantize-> [7, 8, 9, 10, 11, -12]
    let mut result = tensor1.mul(tensor2);  
    result.print();     // ["0x0f5370","0x0f38e0","0x0f44fc","0x0f7e68"] -quantize-> [44, -24, 7, 154]
    assert(result.values==[1004400, 997600, 1000700, 1015400]);
}

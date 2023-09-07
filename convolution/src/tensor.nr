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

    fn set(mut self, index_arr: [comptime Field; S], value: Field) {
        self.values[self.get_index(index_arr)] = value;
    }

    fn quantize_add(self, x: Field, y: Field) -> Field {
        let v = quantize::add(x, self.zero_point, self.scale, y, self.zero_point, self.scale, self.zero_point, self.scale);
        v
    }

    fn quantize_sub(self, x: Field, y: Field) -> Field {
        let v = quantize::sub(x, self.zero_point, self.scale, y, self.zero_point, self.scale, self.zero_point, self.scale);
        v
    }

    fn quantize_mul(self, x: Field, y: Field) -> Field {
        let v = quantize_mul(x, self.zero_point, self.scale, y, self.zero_point, self.scale, self.zero_point, self.scale);
        v
    }

    fn quantize_div(self, x: Field, y: Field) -> Field {
        let v = quantize_div(x, self.zero_point, self.scale, y, self.zero_point, self.scale, self.zero_point, self.scale);
        v
    }

    fn relu(mut self) -> [Field; N] {
        for i in 0..self.values.len() {
            if self.values[i] as u126 < self.zero_point as u126 {
                self.values[i] = self.zero_point;
            }
        }
        self.values
    }

    fn reshape<M>(self, shape: [comptime Field; M]) -> Tensor<N, M> {
        let mut result =  Tensor {values: [0; N], shape: shape, zero_point: self.zero_point, scale: self.scale};
        result
    }

    fn slice<M>(self, start: [comptime Field; S], end: [comptime Field; S]) -> Tensor<M, S> {
        let mut result =  Tensor {values: [0; M], shape: start, zero_point: self.zero_point, scale: self.scale};
        for i in 0..self.shape.len() {
            assert(start[i] as u32<end[i] as u32);
            assert(start[i] as u32<self.shape[i] as u32);
            assert(end[i] as u32<=self.shape[i] as u32);
            result.shape[i] = start[i] - end[i];
        }
        let mut result_index = [0; S];
        let mut index = start;
        for k in 0..start.len() {
            for _ in 0..(start[k] - end[k]) {
                result_index[k] += 1;
                index[k] += 1;
                result.set(result_index, self.get(index));
            }
        }
        result
    }

    // Method to add two matrices
    fn add(self, tensor: Self) -> Self {
        // Check if quantization parameters allow for fast addition
        if (self.zero_point == 0) & (self.scale == 1) & (tensor.zero_point == 0) & (tensor.scale == 1) {
            let values = add(self.values, tensor.values); // Add corresponding values
            Self { values, shape: self.shape, zero_point: self.zero_point, scale: self.scale }
        }
        // Quantized addition
        else {
            let mut values = [0; N]; // Initialize array to store result values
            let tensor1_values = self.values;
            let tensor2_values = tensor.values;
            for i in 0..tensor1_values.len(){
                // Perform quantized addition using the quantize module
                values[i] = quantize::add(tensor1_values[i], self.zero_point, self.scale, tensor2_values[i], tensor.zero_point, tensor.scale, self.zero_point, self.scale);
            }
            Self { values, shape: self.shape, zero_point: self.zero_point, scale: self.scale }
        }
    }

    fn sub(self: Tensor<N, S>, tensor: Self) -> Self {
        if (self.zero_point==0) & (self.scale==1) & (tensor.zero_point==0) & (tensor.scale==1) {
            let values = sub(self.values, tensor.values);
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
        else {
            let mut values = [0; N];
            let tensor1_values = self.values;
            let tensor2_values = tensor.values;
            for i in 0..tensor1_values.len(){
                values[i] = quantize::sub(tensor1_values[i], self.zero_point, self.scale, tensor2_values[i], tensor.zero_point, tensor.scale, self.zero_point, self.scale);
            }
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
    }

    fn hadamard_product<M>(self, tensor: Tensor<N, M>) -> Self {
        if (self.zero_point==0) & (self.scale==1) & (tensor.zero_point==0) & (tensor.scale==1) {
            let values = hadamard_product(self.values, tensor.values);
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
        else {
            let mut values = [0; N];
            let tensor1_values = self.values;
            let tensor2_values = tensor.values;
            for i in 0..tensor1_values.len(){
                values[i] = quantize_mul(tensor1_values[i], self.zero_point, self.scale, tensor2_values[i], tensor.zero_point, tensor.scale, self.zero_point, self.scale);
            }
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
    }

    fn max(self) -> Field {
        let mut v = self.values[0];
        for i in 0..self.values.len(){
            if v as u126 < self.values[i] as u126{
                v = self.values[i];
            }
        }
        v
    }

    fn avg(self) -> Field {
        quantize_div(self.sum(), self.zero_point, self.scale, self.values.len(), 0, 1, self.zero_point, self.scale)
    }

    fn sum(self) -> Field {
        let mut s = self.zero_point;
        for i in 0..self.values.len(){
            s = self.quantize_add(s,self.values[i]);
        }
        s
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

    fn scalar_mul(self, scalar: Field) -> Self {
        // scalar: scalar is a quantized value. zero_point and scale the default value is the same as the tensor
        if (self.zero_point==0) & (self.scale==1) {
            let values = scalar_mul(self.values, scalar);
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
        else {
            let mut values = [0; N];
            for i in 0..N{
                values[i] = quantize_mul(scalar, self.zero_point, self.scale, self.values[i], self.zero_point, self.scale, self.zero_point, self.scale);
            }
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
    }

    fn scalar_div(self, scalar: Field) -> Self {
        // scalar: scalar is a quantized value. zero_point and scale the default value is the same as the tensor
        if (self.zero_point==0) & (self.scale==1) {
            let values = scalar_div(self.values, scalar);
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
        else {
            let mut values = [0; N];
            for i in 0..N{
                values[i] = quantize_div(self.values[i], self.zero_point, self.scale, scalar, self.zero_point, self.scale, self.zero_point, self.scale);
            }
            Self {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
    }

    fn transpose(self: Tensor<N, 2>) -> Tensor<N, 2> {
        assert(self.shape.len()==2);
        let values = transpose(self.values, self.shape);
        Tensor {values, shape: [self.shape[1], self.shape[0]], zero_point: self.zero_point, scale: self.scale}
    }

    fn determinant(self: Tensor<N, 2>) -> Field {
        assert(self.shape.len()==2);
        // Check if the tensor has no scaling or zero-point adjustments
        if (self.zero_point == 0) & (self.scale == 1) {
            let det = determinant(self.values, self.shape);
            det
        } else {
            // Assert that the tensor is square
            assert(self.shape[0] == self.shape[1]);
            
            // Check if the tensor is 2x2
            if self.shape[0] == 2 {
                // Calculate the determinant for a 2x2 tensor with quantized operations
                quantize::sub(
                    quantize_mul(self.values[0], self.zero_point, self.scale, self.values[3], self.zero_point, self.scale, self.zero_point, self.scale),
                    self.zero_point, self.scale,
                    quantize_mul(self.values[1], self.zero_point, self.scale, self.values[2], self.zero_point, self.scale, self.zero_point, self.scale),
                    self.zero_point, self.scale, self.zero_point, self.scale
                )
            } else if self.values[0] == 1 {
                // Return 1 if the tensor is identity-like
                self.values[0]
            } else {
                let mut det = self.zero_point;
                let mut sign = 1;
                
                // Loop through rows/columns to calculate the determinant recursively
                for i in 0..self.shape[0] {
                    let mut minor = [0; N];
                    minor = get_minor(self.values, 0, i, self.shape, minor);
                    let tensor = Tensor {values: minor, shape: [self.shape[0] - 1, self.shape[1] - 1], zero_point: self.zero_point, scale: self.scale};
                    det = quantize::add(
                        det, self.zero_point, self.scale,
                        quantize_mul(
                            quantize_mul(sign, 0, 1, self.values[i], self.zero_point, self.scale, self.zero_point, self.scale),
                            self.zero_point, self.scale,
                            tensor.determinant(),
                            self.zero_point, self.scale, self.zero_point, self.scale
                        ),
                        self.zero_point, self.scale, self.zero_point, self.scale
                    );
                    sign *= -1;
                }
                det
            }
        }
    }

    fn adjoint(self: Tensor<N, 2>) -> Tensor<N, 2> {
        assert(self.shape.len()==2);
        if (self.zero_point == 0) & (self.scale == 1) {
            let values = adjoint(self.values, self.shape);
            Tensor {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        } else {
            assert(self.shape[0] == self.shape[1]);
            let mut values = [0; N];
            
            // Loop through rows and columns to calculate the adjoint tensor
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    let mut minor = [0; N];
                    minor = get_minor(self.values, i, j, self.shape, minor);
                    let tensor = Tensor {values: minor, shape: [self.shape[0] - 1, self.shape[1] - 1], zero_point: self.zero_point, scale: self.scale};
                    let mut sign = 1;
                    for _ in 0..i + j {
                        sign *= -1;
                    }
                    let mut cofactor = quantize_mul(sign, 0, 1, tensor.determinant(), self.zero_point, self.scale, self.zero_point, self.scale);
                    values[i * self.shape[1] + j] = cofactor;
                }
            }
            values = transpose(values, self.shape);
            Tensor {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        }
    }

    fn inverse(self: Tensor<N, 2>) -> Tensor<N, 2> {
        assert(self.shape.len()==2);
        if (self.zero_point == 0) & (self.scale == 1) {
            let values = inverse(self.values, self.shape);
            Tensor {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
        } else {
            assert(self.shape[0] == self.shape[1]);
            let det = self.determinant();
            assert(det != 0);
            let adjoint = self.adjoint();
            let mut values = [0; N];
            
            // Loop through rows and columns to calculate the inverse tensor
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    values[i * self.shape[1] + j] = quantize_div(adjoint.values[i * self.shape[1] + j], self.zero_point, self.scale, det, self.zero_point, self.scale, self.zero_point, self.scale);
                }
            }
            Tensor {values, shape: self.shape, zero_point: self.zero_point, scale: self.scale}
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

fn quantize_div(x: Field, x_z: Field, x_s: Field, y: Field, y_z: Field, y_s: Field, z_z: Field, z_s: Field) -> Field {
    let mut z: Field = (((quantize::div(x, x_z, x_s, y, y_z, y_s, z_z, z_s) * x_s) as u126/(x_s) as u126) as Field);
    z
} 

fn add<M>(tensor1: [Field; M], tensor2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..tensor1.len(){
        result[i] = (tensor1[i] + tensor2[i]);
    }
    result
}

fn sub<M>(tensor1: [Field; M], tensor2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..tensor1.len(){
        result[i] = (tensor1[i] - tensor2[i]);
    }
    result
}

fn hadamard_product<M>(tensor1: [Field; M], tensor2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..tensor1.len(){
        result[i] = (tensor1[i] * tensor2[i]);
    }
    result
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


fn scalar_mul<M>(tensor: [Field; M], scalar: Field) -> [Field; M]{
    // scalar: scalar is a quantized value. zero_point and scale the default value is the same as the tensor
    let mut result = [0; M];
    for i in 0..tensor.len(){
        result[i] = scalar*tensor[i];
    }
    result
}

fn scalar_div<M>(tensor: [Field; M], scalar: Field) -> [Field; M]{
    // scalar: scalar is a quantized value. zero_point and scale the default value is the same as the tensor
    let mut result = [0; M];
    for i in 0..tensor.len(){
        result[i] = tensor[i]/scalar;
    }
    result
}

fn transpose<M>(tensor: [Field; M], shape: [comptime Field; 2]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            result[j * shape[0] + i] = tensor[i * shape[1] + j];
        }
    }
    result
}

fn get_minor<M, N>(tensor: [Field; M], row: Field, col: Field, shape: [comptime Field; 2], mut minor: [Field; N]) -> [Field; N]{
    let mut r: Field = -1;
    for i in 0..shape[0]{
        if i != row{
            r+=1;
            let mut c: Field = -1;
            for j in 0..shape[1]{
                if j != col{
                    c+=1;
                    minor[r*(shape[1]-1)+c]=(tensor[i * shape[1] + j]);
                }
            }
        }
    }
    minor
}

fn determinant<M>(tensor: [Field; M], shape: [comptime Field; 2]) -> Field{
    assert(shape[0] == shape[1]);
    
    if shape[0] == 2{
        tensor[0] * tensor[3] - tensor[1] * tensor[2]
    }
    else if shape[0] == 1{
        tensor[0]
    }
    else {
        let mut det = 0;
        let mut sign = 1;
        for i in 0..shape[0]{
            let mut minor = [0; M];
            minor = get_minor(tensor, 0, i, shape, minor);
            det += sign * tensor[i] * determinant(minor, [shape[0] - 1, shape[1] - 1]);
            sign *= -1;
        }
        det
    }
}

fn adjoint<M>(tensor: [Field; M], shape: [comptime Field; 2]) -> [Field; M]{
    assert(shape[0] == shape[1]);
    let mut adjoint = [0; M];
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            let mut minor = [0; M];
            minor = get_minor(tensor, i, j, shape, minor);
            let mut sign = 1;
            for _ in 0..i+j {
                sign *= -1;
            }
            let mut cofactor = sign * determinant(minor, [shape[0] - 1, shape[1] - 1]);
            adjoint[i*shape[1]+j] = cofactor;
        }
    }
    adjoint = transpose(adjoint, shape);
    adjoint
}

fn inverse<M>(tensor: [Field; M], shape: [comptime Field; 2]) -> [Field; M]{
    assert(shape[0] == shape[1]);
    let det = determinant(tensor, shape);
    assert(det != 0);
    let adjoint = adjoint(tensor, shape);
    let mut inverse = [0; M];
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            inverse[i * shape[1] + j] = adjoint[i * shape[1] + j] / det;
        }
    }
    inverse
}

fn main() {
    let tensor1 = Tensor::new_quantized([999900, 1000200, 1000300, 1000400, 1000500, 999400], [2, 3], 1000000, 100);    // -quantize-> [-1, 2, 3, 4, 5, -6]
    let tensor2 = Tensor::new_quantized([1000700, 1000800, 1000900, 1001000, 1001100, 998800], [3, 2], 1000000, 100);    // -quantize-> [7, 8, 9, 10, 11, -12]
    let mut result = tensor1.mul(tensor2);  
    result.print();     // ["0x0f5370","0x0f38e0","0x0f44fc","0x0f7e68"] -quantize-> [44, -24, 7, 154]
    assert(result.values==[1004400, 997600, 1000700, 1015400]);
}

#[test]
fn test_relu() {
    let tensor1 = Tensor::new_quantized([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 5, 1);    
    let result = tensor1.relu();  
    println(result);     
    assert(tensor1.values==[1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_slice() {
    let tensor1 = Tensor::new_quantized([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 0, 1);   
    let mut result = tensor1.slice([1,1], [3, 3]);  
    println(result);   
    assert(result.values==[5, 6, 8, 9]);
}

#[test]
fn test_add() {
    let tensor1 = [1; 9];
    let tensor2 = [2; 9];
    let result = add(tensor1, tensor2);
    println(result);
    assert(result==[3; 9]);
}

#[test]
fn test_add1() {
    let tensor1 = Tensor::new([1; 9], [3, 3]);
    let tensor2 = Tensor::new([2; 9], [3, 3]);
    let result = tensor1.add(tensor2);  
    result.print();
}

#[test]
fn test_add2() {
    let tensor1 = Tensor::new_quantized([1; 9], [3, 3], 0, 1);
    let tensor2 = Tensor::new_quantized([2; 9], [3, 3], 0, 2);
    let result = tensor1.add(tensor2);
    println(result);  // {"scale":"0x01","shape":["0x03","0x03"],"values":["0x05","0x05","0x05","0x05","0x05","0x05","0x05","0x05","0x05"],"zero_point":"0x00"}
}

#[test]
fn test_sub() {
    let tensor1 = [3; 9];
    let tensor2 = [2; 9];
    let result = sub(tensor1, tensor2);
    println(result);
}

#[test]
fn test_hadamard_product() {
    let tensor1 = [3; 9];
    let tensor2 = [2; 9];
    let result = hadamard_product(tensor1, tensor2);
    println(result);
}

#[test]
fn test_hadamard_product1() {
    let tensor1 = Tensor::new_quantized([4; 9], [3, 3], 0, 2);
    let tensor2 = Tensor::new_quantized([2; 9], [3, 3], 0, 4);
    let result = tensor1.hadamard_product(tensor2);
    println(result);    // {"scale":"0x02","shape":["0x03","0x03"],"values":["0x02","0x02","0x02","0x02","0x02","0x02","0x02","0x02","0x02"],"zero_point":"0x00"}
}

#[test]
fn test_mul() {
    let tensor1 = [1, 2, 3, 4, 5, 6];
    let tensor2 = [7, 8, 9, 10, 11, 12];
    let tensor1_shape = [2,3];
    let tensor2_shape = [3,2];
    let mut result = mul(tensor1, tensor2, tensor1_shape, tensor2_shape);
    println(result);    //  [58, 64, 139, 154] ['0x3a', '0x40', '0x8b', '0x9a']
}

#[test]
fn test_mul1() {
    let tensor1 = Tensor::new_quantized([999900, 1000200, 1000300, 1000400, 1000500, 999400], [2, 3], 1000000, 100);    // -quantize-> [-1, 2, 3, 4, 5, -6]
    let tensor2 = Tensor::new_quantized([1000700, 1000800, 1000900, 1001000, 1001100, 998800], [3, 2], 1000000, 100);    // -quantize-> [7, 8, 9, 10, 11, -12]
    let mut result = tensor1.mul(tensor2);  
    result.print();     // ["0x0f5370","0x0f38e0","0x0f44fc","0x0f7e68"] -quantize-> [44, -24, 7, 154]
    assert(result.values==[1004400, 997600, 1000700, 1015400]);
}

#[test]
fn test_scalar_mul() {
    let tensor = [1, 2, 3, 4, 5, 6];
    let scalar = 2;
    let result = scalar_mul(tensor, scalar);
    println(result);
}

#[test]
fn test_scalar_mul1() {
    let tensor1 = Tensor::new_quantized([4; 9], [3, 3], 0, 2);
    let scalar = 4;     // scalar is a quantized value. zero_point and scale the default value is the same as the tensor
    let result = tensor1.scalar_mul(scalar);
    println(result);    // {"scale":"0x02","shape":["0x03","0x03"],"values":["0x08","0x08","0x08","0x08","0x08","0x08","0x08","0x08","0x08"],"zero_point":"0x00"}
}

#[test]
fn test_scalar_div() {
    let tensor = [2, 4, 6];
    let scalar = 2;
    let result = scalar_div(tensor, scalar);
    println(result);
}

#[test]
fn test_transpose() {
    let tensor = [1, 2, 3, 4, 5, 6];
    let shape = [2, 3];
    let result = transpose(tensor, shape);
    println(result);    // [1, 4, 2, 5, 3, 6] ['0x1', '0x4', '0x2', '0x5', '0x3', '0x6']
}

#[test]
fn test_transpose1() {
    let tensor1 = Tensor::new([1, 2, 3, 4, 5, 6], [2, 3]);    // -quantize-> [-1, 2, 3, 4, 5, -6]
    let result = tensor1.transpose(); 
    println(result);     // {"scale":"0x01","shape":["0x03","0x02"],"values":["0x01","0x04","0x02","0x05","0x03","0x06"],"zero_point":"0x00"}
    let transpose = Tensor::new([1, 4, 2, 5, 3, 6], [3, 2]);
    assert(result.shape==transpose.shape);
    assert(result.values==transpose.values);
}

#[test]
fn test_determinant() {
    let tensor = [0, 1, 2, 4, 5, 6, 7, 8, 0];
    let shape = [3, 3];
    let result = determinant(tensor, shape);
    println(result);    // 36 '0x24'
}

#[test]
fn test_determinant1() {
    let tensor1 = Tensor::new_quantized([12, 12, 13, 14, 15, 16, 17, 18, 19], [3, 3], 10, 1);    // -quantize-> [2, 2, 3, 4, 5, 6, 7, 8, 9]
    let result = tensor1.determinant(); 
    println(result);     // 0x07 -quantize-> -3
    assert(result==7);
}

#[test]
fn test_adjoint() {
    let tensor = [0, 1, 2, 4, 5, 6, 7, 8, 0];
    let shape = [3, 3];
    let result = adjoint(tensor, shape);
    println(result);    // ['-0x30', '0x10', '-0x4', '0x2a', '-0xe', '0x8', '-0x3', '0x7', '-0x4']
}

#[test]
fn test_adjoint1() {
    let tensor1 = Tensor::new_quantized([12, 12, 13, 14, 15, 16, 17, 18, 19], [3, 3], 10, 1);    // -quantize-> [2, 2, 3, 4, 5, 6, 7, 8, 9]
    let result = tensor1.adjoint(); 
    result.print();     // ["0x07","0x10","0x07","0x10","0x07","0x0a","0x07","0x08","0x0c"] -quantize-> [-3, 6, -3, 6, -3, 0, -3, -2, 2]
    assert(result.values==[7, 16, 7, 16, 7, 10, 7, 8, 12]);
}

#[test]
fn test_inverse() {
    let tensor = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    let shape = [3, 3];
    let result = inverse(tensor, shape);
    println(result);    // ['-0x30', '0x10', '-0x4', '0x2a', '-0xe', '0x8', '-0x3', '0x7', '-0x4']
}

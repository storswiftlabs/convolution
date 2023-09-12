mod tensor;
mod conv;
use tensor::Tensor;
use conv::Linear;
use dep::std::println;

global in_len = 10;
global w1_len = 100;
global b1_len = 10;
global zero_point = 100000;
global scale = 10000;

fn main(inputs: [Field; in_len], w1 : [Field; w1_len], b1 : [Field; b1_len]) {
    let weight1 = Tensor::new_quantized(w1, [10, 10], zero_point, scale);
    let bias1 = Tensor::new_quantized(b1, [1, 10], zero_point, scale);
    // Create a linear layer
    let linear_layer = Linear::new(weight1, bias1);

    // Perform forward pass
    let input = Tensor::new_quantized(inputs, [1, 10], zero_point, scale);
    let linear_output = linear_layer.forward(input);
    println(linear_output);
}

#[test]
fn test_main() {
    let inputs = [0; in_len];
    let w1 = [0; w1_len];
    let b1 = [0; b1_len];
    main(inputs, w1, b1);
}

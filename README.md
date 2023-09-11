# Convolution

This is a simple implementation of a convolutional neural network (CNN) library in Noir. The library provides basic building blocks for creating and working with CNN layers, including Convolutional layers, Pooling layers, and Linear (fully connected) layers.

## Features

- Convolutional Layer: The library includes a `Conv2D` struct for creating convolutional layers. You can specify the weight, bias, stride, and padding parameters when creating a convolutional layer. The layer supports both forward and padding operations.

- Pooling Layer: The library includes a `Pool2D` struct for creating pooling layers. You can specify the pooling type (max or average), stride, padding, and kernel size when creating a pooling layer. The layer supports both max pooling and average pooling operations.

- Linear (Fully Connected) Layer: The library includes a `Linear` struct for creating linear layers. You can specify the weight and bias when creating a linear layer. The layer supports forward operations.

## Usage

Here's a basic example of how to use the library to create a simple CNN:

```rust
use dep::convolution::{conv::{Conv2D, Pool2D, Linear}, tensor::Tensor};

global in_len = 784;
global w1_len = 45;
global b1_len = 5;
global w2_len = 90;
global b2_len = 10;
global w3_len = 490;
global b3_len = 10;
global zero_point = 100000;
global scale = 10000;


fn main(inputs: [Field; in_len], w1 : [Field; w1_len], b1 : [Field; b1_len], w2 : [Field; w2_len], b2 : [Field; b2_len], w3 : [Field; w3_len], b3 : [Field; b3_len]) {

    let weight1 = Tensor::new_quantized(w1, [5, 1, 3, 3], zero_point, scale);
    let bias1 = Tensor::new_quantized(b1, [5], zero_point, scale);
    let weight2 = Tensor::new_quantized(w2, [10, 1, 3, 3], zero_point, scale);
    let bias2 = Tensor::new_quantized(b2, [10], zero_point, scale);
    let weight3 = Tensor::new_quantized(w3, [49, 10], zero_point, scale);
    let bias3 = Tensor::new_quantized(b3, [10], zero_point, scale);

    // Create a convolutional layer
    let conv_layer1 = Conv2D::new(weight1, bias1, 1, 1);
    let conv_layer2 = Conv2D::new(weight2, bias2, 1, 1);

    // Create a pooling layer
    let pooling_layer = Pool2D {
        stride: 2,
        padding: 0,
        kernel_size: 2,
    };

    // Create a linear layer
    let linear_layer = Linear::new(weight3, bias3);

    // Perform forward pass
    let input = Tensor::new_quantized(inputs, [1, 28, 28], zero_point, scale);
    let mut conv_output1 = conv_layer1.forward(input);
    conv_output1.values = conv_output1.relu();
    let pooled_output1 = pooling_layer.max_pooling(conv_output1);
    let mut conv_output2 = conv_layer2.forward(pooled_output1);
    conv_output2.values = conv_output2.relu();
    let pooled_output2 = pooling_layer.max_pooling(conv_output2);
    // Fixed the side with 1 to have only one column
    let reshape_output = pooled_output2.reshape([1, weight3.shape[0]]);
    let linear_output = linear_layer.forward(reshape_output);
}

#[test]
fn test_main() {
    let inputs = [0; in_len];
    let w1 = [0; w1_len];
    let b1 = [0; b1_len];
    let w2 = [0; w2_len];
    let b2 = [0; b2_len];
    let w3 = [0; w3_len];
    let b3 = [0; b3_len];
    main(inputs, w1, b1, w2, b2, w3, b3);
}

```

## Dependencies

This library utilizes [quantized_arithmetic](https://github.com/storswiftlabs/quantized_arithmetic/tree/main) for quantized arithmetic operations and employs [matrix_operations](https://github.com/storswiftlabs/matrix_operations) for matrix operations.

## License

This project  is provided under the [Apache License, Version 2.0](LICENSE).

## Acknowledgments

This library is a simple implementation and may not cover all the features and optimizations required for a production-level CNN library. It serves as a starting point for building more complex neural networks in Noir.

If you have any questions, suggestions, or contributions, please feel free to contact the project maintainers.
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    // e^{-15} ≈ 3.059 * 10^{-7}

    let sig: f64 = 1.0 / (1.0 + E.powf(-x));
    return sig;
}

//Think of it as a neuron
fn feedforward(x: &Vec<f64>, w: &Vec<f64>, b: &f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * w[i];
    }

    let result = sigmoid(sum + b);

    result
}

fn train(data: &Vec<f64>, w: &Vec<f64>, b: &f64) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];

    for _ in 0..2 {
        result.push(feedforward(&data, &w, &b));
    }

    println!("result-->: {:#?}", result);
    return result;
}

fn _prediction_test() {
    let height = 170.0;
    let weight = 65.0;
    let x = vec![height, weight];

    let w = vec![0.5, -0.5];
    let b = 0.0;

    let prediction = feedforward(&x, &w, &b);

    if prediction >= 0.5 {
        println!("Man: ({})", prediction);
    } else {
        println!("Woman: ({})", prediction);
    }
}

fn main() {
    let x = vec![170.0, 65.0]; //inputs
    let w = vec![0.5, 0.5]; //weights
    let b = 0.0; //bias

    let data = train(&x, &w, &b);

    println!("DAta-->: {:#?}", data);
    let o1 = feedforward(&data, &w, &b);

    // let h1 = feedforward(&x, &w, &b);
    // let h2 = feedforward(&x, &w, &b);
    // let o1 = feedforward(&vec![h1, h2], &w, &b);

    println!("Output: {}", o1);
}

#[cfg(test)]

mod test {

    use super::*;

    // #[test]
    // fn nn_test() {
    //     let x = vec![4.0, 15.0]; //inputs
    //     let w = vec![0.0, 1.0]; //weights
    //     let b = 0.0; //bias

    //     let prediction = feedforward(&x, &w, b);
    //     let expected = (prediction - 0.999999694097773).abs();

    //     assert!(expected < 1e-6, "Prediction: {:#?} ", prediction);
    // }

    // #[test]
    // fn test_feedforwad() {
    //     let x = vec![4.0, 15.0]; //inputs
    //     let w = vec![0.0, 1.0]; //weights
    //     let b = 0.0; //bias
    //     let prediction = feedforward(x, w, b);
    //     let expected = (prediction - 0.999999694097773).abs();
    //     assert!(expected < 1e-6, "Prediction: {:#?} ", prediction);
    // }

    #[test]
    fn test_sigmoid_should_pass() {
        let result = sigmoid(15.0); // 0.999999694097773
        let expected = (result - 0.999999694097773).abs();

        // to make it more sensitive you 1e-7 or 1e-9
        let tolerance = 1e-6;

        assert!(
            expected < tolerance,
            "Sigmoid(15) not close to the expected value: {}",
            result
        );
    }

    #[test]
    #[should_panic]
    fn test_sigmoid_should_fail() {
        let result = sigmoid(11.0);
        let expected = (result - 0.999999694097773).abs();
        let tolerance = 1e-6;
        assert!(
            expected < tolerance,
            "Sigmoid(11) not close to the expected (0) value: {}",
            result
        );
    }
}

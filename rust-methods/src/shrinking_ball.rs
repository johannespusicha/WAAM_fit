use crate::linear_algebra::Vector3D;

fn extent(points: Vec<[f64; 3]>) -> f64 {
    let x_max = points
        .iter()
        .fold(f64::NEG_INFINITY, |a, [b, _, _]| a.max(*b));
    let x_min = points.iter().fold(f64::INFINITY, |a, [b, _, _]| a.min(*b));
    let y_max = points
        .iter()
        .fold(f64::NEG_INFINITY, |a, [_, b, _]| a.max(*b));
    let y_min = points.iter().fold(f64::INFINITY, |a, [_, b, _]| a.min(*b));
    let z_max = points
        .iter()
        .fold(f64::NEG_INFINITY, |a, [_, _, b]| a.max(*b));
    let z_min = points.iter().fold(f64::INFINITY, |a, [_, _, b]| a.min(*b));

    ((x_max - x_min).powi(2) + (y_max - y_min).powi(2) + (z_max - z_min).powi(2)).sqrt()
}

#[cfg(test)]
mod shrinking_ball_test {
    use super::*;

    #[test]
    fn test_extent() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]];

        assert_eq!(extent(points), 3.0_f64.sqrt());
    }
}

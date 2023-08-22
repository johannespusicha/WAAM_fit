use std::collections::HashMap;

use crate::linear_algebra::Vector3D;
use kdtree::{distance::squared_euclidean, KdTree};

#[derive(Debug)]
struct BrepElement {
    point: Vector3D,
    normal: Vector3D,
}

#[derive(Debug)]
pub struct TreeManager3D {
    data: KdTree<f64, usize, [f64; 3]>,
    index: HashMap<usize, BrepElement>,
    extent: f64,
}

impl TreeManager3D {
    pub fn from_points_and_normals(
        points: Vec<Vector3D>,
        normals: Vec<Vector3D>,
        indices: Vec<u64>,
    ) -> Self {
        assert!(points.len() == normals.len(), "Length of points and normals did not match. Since they are corresponding data, they are required to have equal size.");
        let mut tree: KdTree<f64, usize, [f64; 3]> = KdTree::with_capacity(3, 64);
        let mut map = HashMap::new();
        let extent = extent(&points);
        for (index, (point, normal)) in indices.iter().zip(points.iter().zip(normals.iter())) {
            let index = *index as usize;
            if let Some(err) = tree.add(point.to_array(), index).err() {
                println!("{err}");
            }
            map.insert(
                index,
                BrepElement {
                    point: *point,
                    normal: *normal,
                },
            );
        }

        TreeManager3D {
            data: tree,
            index: map,
            extent,
        }
    }

    fn lim_nearest_but(
        &self,
        point: &Vector3D,
        but: &Vector3D,
        max_distance: f64,
    ) -> Option<Vector3D> {
        assert!(
            self.data.size() >= 2,
            "Did not find enough data in the tree. At least two points are needed."
        );
        let neighbours = self.data.nearest(&point.to_array(), 2, &squared_euclidean);
        match neighbours {
            Ok(neighbours) => {
                for (distance, index) in neighbours {
                    let nearest = self.index.get(index).unwrap().point;
                    if (max_distance - distance.sqrt()) > 1.0e-6 && &nearest != but {
                        return Some(nearest);
                    }
                }
                // else: Ball is empty.
                None
            }
            Err(msg) => {
                println!("{msg}");
                None
            }
        }
    }
}

fn extent(points: &[Vector3D]) -> f64 {
    let x_max = points
        .iter()
        .map(|v| v.i)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let x_min = points
        .iter()
        .map(|v| v.i)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let y_max = points
        .iter()
        .map(|v| v.j)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let y_min = points
        .iter()
        .map(|v| v.j)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let z_max = points
        .iter()
        .map(|v| v.k)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let z_min = points
        .iter()
        .map(|v| v.k)
        .fold(f64::INFINITY, |a, b| a.min(b));

    ((x_max - x_min).powi(2) + (y_max - y_min).powi(2) + (z_max - z_min).powi(2)).sqrt()
}

/// Calculates the medial radius for a given base point and normal by applying the shrinking ball algorithm.
///
///
/// # Error
/// Since this is an iterative algortihm, the maximum number of iterations is bound to 100.
/// Gives back an ´Err´ in case that the maximum number of iterations is reached and no solution was found.
///
/// # Acknowledgements
/// The shrinking ball algorithm was originally introduced by [^ma12]
///
/// [^ma12]: Ma, J., Bae, S.W. & Choi, S. 3D medial axis point approximation using nearest neighbors and the normal field. Vis Comput 28, 7–19 (2012). https://doi.org/10.1007/s00371-011-0594-7
pub fn shrink_ball(
    base: &Vector3D,
    normal: &Vector3D,
    tree: &TreeManager3D,
) -> Result<f64, String> {
    let normal_unit = *normal * (1.0 / normal.length());
    let extent = 2.0 * tree.extent;
    let mut radius = extent;

    let mut remaining = 10;
    while remaining > 0 {
        assert!(radius >= 0.0, "Expected radius to be >= 0");
        assert!(radius < f64::INFINITY);
        remaining -= 1;

        let center = base + normal_unit * radius;
        match tree.lim_nearest_but(&center, base, radius) {
            Some(nearest) => {
                // Point was contained in ball: Calc radius for a new ball on normal which touches base and near:
                let base_to_near = nearest - base;
                let next_radius =
                    0.5 * base_to_near.dot(&base_to_near) / base_to_near.dot(&normal_unit);
                radius = next_radius;
            }
            //Termination condition: Ball is empty
            None => {
                if radius == extent {
                    // radius > extent of geometry => No restriction to radius.
                    return Ok(f64::INFINITY);
                } else {
                    return Ok(radius);
                }
            }
        }
    }
    // In the case that all remaining cycles are used up before a solution was found:
    Err(format!(
        "Iteration did not converge. Giving up on this point:  {:?}",
        base
    ))
}

impl TreeManager3D {
    pub fn eval_radii(&self) -> Vec<f64> {
        let mut radii = vec![];

        for (index, element) in &self.index {
            let radius = match shrink_ball(&element.point, &element.normal, self) {
                Err(msg) => {
                    println!("{}", msg);
                    -1.0
                }
                Ok(radius) => radius,
            };
            radii.push((index, radius));
        }

        radii.sort_unstable_by_key(|tuple| tuple.0);
        radii.iter().map(|tuple| tuple.1).collect()
    }
}

#[cfg(test)]
mod shrinking_ball_test {
    use super::*;

    #[test]
    fn test_extent() {
        let points = vec![
            Vector3D::new(0, 0, 0),
            Vector3D::new(1, 1, 1),
            Vector3D::new(1, 0, 0),
        ];
        assert_eq!(extent(&points), 3.0_f64.sqrt());
    }
}
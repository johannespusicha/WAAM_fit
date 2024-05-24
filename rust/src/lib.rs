use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

mod linear_algebra;
mod shrinking_ball;
use linear_algebra::Vector3D;
use shrinking_ball::*;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn get_sphere_radii<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<f64>,
    normals: PyReadonlyArray2<f64>,
    indices: Vec<u64>,
    #[cfg(feature = "includeBaseplate")] base: ((f64, f64, f64), (f64, f64, f64)),
) -> PyResult<(Vec<usize>, &'py PyArray2<f64>)> {
    if centers.len() != normals.len() {
        return Err(PyValueError::new_err(format!("The node_coordinates and normal_vectors matrices must have the same length. Found: {:?} and {:?}", centers.len(), normals.len())));
    }
    let centers = centers.as_array();
    let centers: Vec<Vector3D> = centers
        .rows()
        .into_iter()
        .map(|row| Vector3D::new(row[0], row[1], row[2]))
        .collect();
    let normals = normals.as_array();
    let normals: Vec<Vector3D> = normals
        .rows()
        .into_iter()
        .map(|row| Vector3D::new(row[0], row[1], row[2]))
        .collect();
    #[cfg(not(feature = "includeBaseplate"))]
    {
        let tree = TreeManager3D::from_points_and_normals(centers, normals, indices);

        Ok(tree.eval_radii())
    }

    #[cfg(feature = "includeBaseplate")]
    {
        let base_a = Vector3D::from(base.0);
        let base_b = Vector3D::from(base.1);
        let tree =
            TreeManager3D::from_points_and_normals(centers, normals, indices, (base_a, base_b));
        println!("Info\t: Done partitioning k-d-tree.");
        println!("Info\t: {}", tree.get_stats());

        let (tags, raw_data) = tree.eval_radii();

        Ok((tags, raw_data.into_pyarray(py)))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_methods(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_sphere_radii, m)?)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Vector3D {
    i: f64,
    j: f64,
    k: f64,
}

impl Vector3D {
    pub fn new(i: impl Into<f64>, j: impl Into<f64>, k: impl Into<f64>) -> Self {
        Vector3D {
            i: i.into(),
            j: j.into(),
            k: k.into(),
        }
    }

    pub fn length(&self) -> f64 {
        (self.i * self.i + self.j * self.j + self.k * self.k).sqrt()
    }
}
#[cfg(test)]
mod brep_element_tests {
    use super::*;


    #[test]
    fn test_create_vector_from_non_f64() {
        assert_eq!(
            Vector3D::new(1, 2, 3),
            Vector3D {
                i: 1.0,
                j: 2.0,
                k: 3.0
            }
        )
    }

    #[test]
    fn test_vector_length() {
        let v = Vector3D::new(2.0, 3.0, 4.0);
        assert_eq!(v.length(), 29.0_f64.sqrt())
    }
}

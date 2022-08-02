use ndarray;
use numpy::ToPyArray;
use pyo3::prelude::*;

fn assemble_global_matrix(global_mtx: &mut ndarray::Array2<f64>, local_matrices: &ndarray::Array3<f64>, elements: &ndarray::Array2<i64>) {
    for (idx,lm) in local_matrices.outer_iter().enumerate() {
        let m = lm.shape()[0];
        for i in 0..m {
            for j in 0..m {
                let first_pos = elements[[idx, i]] as usize;
                let second_pos = elements[[idx, j]] as usize;
                global_mtx[[first_pos, second_pos]] += lm[[i,j]];
            }
        }
    }
}

fn assemble_global_vector(global_vector: &mut ndarray::Array1<f64>, local_vectors: &ndarray::Array2<f64>, elements: &ndarray::Array2<i64>) {
    for (idx,lv) in local_vectors.outer_iter().enumerate() {
        let m = lv.shape()[0];
        for i in 0..m {
            let pos = elements[[idx, i]] as usize;
            global_vector[[pos]] += lv[[i]];
        }
    }
}

#[pymodule]
#[pyo3(name = "fem_toolkit")]
fn fem_toolkit(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "assemble_global_matrix")]
        fn assemble_global_matrix_py<'py>(_py: Python<'py>, 
                                global_mtx: &numpy::PyArray2<f64>,
                                local_matrices: &numpy::PyArray3<f64>,
                                elements: &numpy::PyArray2<i64>) -> &'py numpy::PyArray2<f64>
    {
        let global_mtx = unsafe { global_mtx.as_array_mut() };
        let  mut global_mtx = global_mtx.to_owned();
        let local_matrices = unsafe { local_matrices.as_array_mut() };
        let local_matrices = local_matrices.to_owned();
        let elements = unsafe { elements.as_array_mut() };
        let elements = elements.to_owned();

        assemble_global_matrix(&mut global_mtx, &local_matrices, &elements);

        global_mtx.to_pyarray(_py)
    }

    #[pyfn(m)]
    #[pyo3(name = "assemble_global_vector")]
    fn assemble_global_vector_py<'py>(_py: Python<'py>, 
                        global_vector: &numpy::PyArray1<f64>,
                        local_vectors: &numpy::PyArray2<f64>,
                        elements: &numpy::PyArray2<i64>) -> &'py numpy::PyArray1<f64>
    {
        let global_vector = unsafe { global_vector.as_array_mut() };
        let  mut global_vector = global_vector.to_owned();
        let local_vectors = unsafe { local_vectors.as_array_mut() };
        let local_vectors = local_vectors.to_owned();
        let elements = unsafe { elements.as_array_mut() };
        let elements = elements.to_owned();

        assemble_global_vector(&mut global_vector, &local_vectors, &elements);

        global_vector.to_pyarray(_py)
    }

    Ok(())
}
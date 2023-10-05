use crate::common;
use pyo3::exceptions;
use pyo3::prelude::*;

#[pyfunction]
pub fn hill_diversity(class_counts: Vec<u32>, q: f32) -> PyResult<f32> {
    /*
    Compute Hill diversity.

    Hill numbers - express actual diversity as opposed e.g. to Gini-Simpson (probability) and Shannon (information)

    exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    Ssee "Entropy and diversity" by Lou Jost

    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    */
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Please select a non-zero value for q.",
        ));
    }
    let num: u32 = class_counts.iter().sum();
    // catch potential division by zero situations
    let mut hill = 0.0;
    if num != 0 {
        // hill number defined in the limit as the exponential of information entropy
        if (q - 1.0).abs() < f32::EPSILON {
            let mut ent = 0.0;
            for class_count in class_counts {
                if class_count != 0 {
                    // if not 0
                    let prob = class_count as f32 / num as f32; // the probability of this class
                    ent += prob * prob.log(std::f32::consts::E); // sum entropy
                }
            }
            hill = (-ent).exp(); // return exponent of entropy
        }
        // otherwise use the usual form of Hill numbers
        else {
            let mut div = 0.0;
            for class_count in class_counts {
                if class_count != 0 {
                    let prob = class_count as f32 / num as f32; // the probability of this class
                    div += prob.powf(q); // sum
                }
            }
            hill = div.powf(1.0 / (1.0 - q)); // return as equivalent species
        }
    }
    Ok(hill)
}

#[pyfunction]
pub fn hill_diversity_branch_distance_wt(
    class_counts: Vec<u32>,
    class_distances: Vec<f32>,
    q: f32,
    beta: f32,
    max_curve_wt: f32,
) -> PyResult<f32> {
    /*
    Compute Hill diversity weighted by branch distances.

    Based on unified framework for species diversity in Chao, Chiu, Jost 2014.
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    */
    if class_counts.len() != class_distances.len() {
        return Err(exceptions::PyValueError::new_err(
            "Mismatching number of unique class counts and respective class distances.",
        ));
    }
    if beta < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Please provide the beta without the leading negative.",
        ));
    }
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Please select a non-zero value for q.",
        ));
    }
    // catch potential division by zero situations
    let num: u32 = class_counts.iter().sum();
    if num == 0 {
        return Ok(0.0);
    }
    // find T
    let mut agg_t = 0.0;
    for (class_count, class_dist) in class_counts.iter().zip(class_distances.iter()) {
        if *class_count != 0 {
            let proportion = *class_count as f32 / num as f32;
            let wt = common::clipped_beta_wt(beta, max_curve_wt, *class_dist)?;
            agg_t += wt * proportion;
        }
    }
    // hill number defined in the limit as the exponential of information entropy
    if (q - 1.0).abs() < 1.0e-7 {
        let mut div_branch_wt_lim = 0.0;
        // get branch lengths and class abundances
        for (class_count, class_dist) in class_counts.iter().zip(class_distances.iter()) {
            if *class_count != 0 {
                let proportion = *class_count as f32 / num as f32;
                let wt = common::clipped_beta_wt(beta, max_curve_wt, *class_dist)?;
                div_branch_wt_lim += wt * proportion / agg_t * (proportion / agg_t).ln();
                // sum entropy
            }
        }
        // return exponent of entropy
        return Ok((-div_branch_wt_lim).exp());
    } else {
        // otherwise use the usual form of Hill numbers
        let mut div_branch_wt = 0.0;
        // get branch lengths and class abundances
        for (class_count, class_dist) in class_counts.iter().zip(class_distances.iter()) {
            if *class_count != 0 {
                let a = *class_count as f32 / num as f32;
                let wt = common::clipped_beta_wt(beta, max_curve_wt, *class_dist)?;
                div_branch_wt += wt * (a / agg_t).powf(q);
            }
        }
        // once summed, apply q
        return Ok(div_branch_wt.powf(1.0 / (1.0 - q)));
    }
}

#[pyfunction]
pub fn hill_diversity_pairwise_distance_wt(
    class_counts: Vec<u32>,
    class_distances: Vec<f32>,
    q: f32,
    beta: f32,
    max_curve_wt: f32,
) -> PyResult<f32> {
    /*
    Compute Hill diversity weighted by pairwise distances.

    This is the distances version - see below for disparity matrix version

    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k
    Remember these are already distilled species counts - so it is OK to use closest distance to each species

    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i
    */
    if class_counts.len() != class_distances.len() {
        return Err(exceptions::PyValueError::new_err(
            "Mismatching number of unique class counts and respective class distances.",
        ));
    }
    if beta < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Please provide the beta without the leading negative.",
        ));
    }
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Please select a non-zero value for q.",
        ));
    }
    // catch potential division by zero situations
    let num: u32 = class_counts.iter().sum();
    if num == 0 {
        return Ok(0.0);
    }
    // calculate Q
    let mut agg_q = 0.0;
    for (i, &class_count_i) in class_counts.iter().enumerate() {
        if class_count_i == 0 {
            continue;
        }
        let a_i = class_count_i as f32 / num as f32;
        for (j, &class_count_j) in class_counts.iter().enumerate() {
            // only need to examine the pair if j < i, otherwise double-counting
            if j > i {
                break;
            }
            if class_count_j == 0 {
                continue;
            }
            let a_j = class_count_j as f32 / num as f32;
            let wt = common::clipped_beta_wt(
                beta,
                max_curve_wt,
                class_distances[i] + class_distances[j],
            )?;
            // pairwise distances
            agg_q += wt * a_i * a_j;
        }
    }
    // pairwise disparities weights can sometimes give rise to Q = 0... causing division by zero etc.
    if agg_q.abs() < std::f32::EPSILON {
        return Ok(0.0);
    }
    // if in the limit, use exponential
    if (q - 1.0).abs() < std::f32::EPSILON {
        let mut div_pw_wt_lim = 0.0;
        for (i, &class_count_i) in class_counts.iter().enumerate() {
            if class_count_i == 0 {
                continue;
            }
            let a_i = class_count_i as f32 / num as f32;
            for (j, &class_count_j) in class_counts.iter().enumerate() {
                // only need to examine the pair if j < i, otherwise double-counting
                if j > i {
                    break;
                }
                if class_count_j == 0 {
                    continue;
                }
                let a_j = class_count_j as f32 / num as f32;
                // pairwise distances
                let wt = common::clipped_beta_wt(
                    beta,
                    max_curve_wt,
                    class_distances[i] + class_distances[j],
                )?;
                div_pw_wt_lim +=
                    wt * a_i * a_j / agg_q * ((a_i * a_j) / agg_q).log(std::f32::consts::E);
            }
        }
        // once summed
        // (FD_lim / Q) ** (1 / 2)
        return Ok(((-div_pw_wt_lim).exp() as f64).sqrt() as f32);
    } else {
        // otherwise conventional form
        let mut div_pw_wt = 0.0;
        for (i, &class_count_i) in class_counts.iter().enumerate() {
            if class_count_i == 0 {
                continue;
            }
            let a_i = class_count_i as f32 / num as f32;
            for (j, &class_count_j) in class_counts.iter().enumerate() {
                // only need to examine the pair if j < i, otherwise double-counting
                if j > i {
                    break;
                }
                if class_count_j == 0 {
                    continue;
                }
                let a_j = class_count_j as f32 / num as f32;
                // pairwise distances
                let wt = common::clipped_beta_wt(
                    beta,
                    max_curve_wt,
                    class_distances[i] + class_distances[j],
                )?;
                div_pw_wt += wt * ((a_i * a_j / agg_q).powf(q));
            }
        }
        // (FD / Q) ** (1 / 2)
        return Ok((div_pw_wt.powf(1.0 / (1.0 - q)) as f64).sqrt() as f32);
    }
}

#[pyfunction]
pub fn gini_simpson_diversity(class_counts: Vec<u32>) -> PyResult<f32> {
    /*
    Gini-Simpson diversity.
    Gini transformed to 1 − λ
    Probability that two individuals picked at random do not represent the same species (Tuomisto)
    Ordinarily:
    D = 1 - sum(p**2) where p = Xi/N
    Bias corrected:
    D = 1 - sum(Xi/N * (Xi-1/N-1))
    */
    let num = class_counts.iter().sum::<u32>();
    // catch potential division by zero situations
    if num < 2 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;
    let num_minus_1_f32 = (num - 1) as f32;
    // compute bias corrected gini-simpson
    let gini: f32 = class_counts
        .iter()
        .map(|&x| {
            let x_f32 = x as f32;
            x_f32 / num_f32 * ((x_f32 - 1.0) / num_minus_1_f32)
        })
        .sum();
    Ok(1.0 - gini)
}

#[pyfunction]
pub fn shannon_diversity(class_counts: Vec<u32>) -> PyResult<f32> {
    /*
    Shannon diversity (information entropy).
    Entropy
    p = Xi/N
    S = -sum(p * log(p))
    Uncertainty of the species identity of an individual picked at random (Tuomisto)
    */
    let num: u32 = class_counts.iter().sum();
    let mut shannon: f32 = 0.0;
    // catch potential division by zero situations
    if num == 0 {
        return Ok(shannon);
    }
    // compute
    for class_count in class_counts {
        if class_count > 0 {
            let prob = class_count as f32 / num as f32; // the probability of this class
            shannon += prob * prob.log(std::f32::consts::E); // sum entropy
        }
    }
    // remember negative
    Ok(-shannon)
}

#[pyfunction]
pub fn raos_quadratic_diversity(
    class_counts: Vec<u32>,
    wt_matrix: Vec<Vec<f32>>,
    alpha: f32,
    beta: f32,
) -> PyResult<f32> {
    /*
    Rao's quadratic diversity.

    Bias corrected and based on disparity

    Sum of weighted pairwise products

    Note that Stirling's diversity is a rediscovery of Rao's quadratic diversity
    Though adds alpha and beta exponents to tweak weights of disparity dij and pi * pj, respectively
    This is a hybrid of the two, i.e. including alpha and beta options and adjusted for bias
    Rd = sum(dij * Xi/N * (Xj/N-1))

    Behaviour is controlled using alpha and beta exponents
    0 and 0 reduces to variety (effectively a count of unique types)
    0 and 1 reduces to balance (half-gini - pure balance, no weights)
    1 and 0 reduces to disparity (effectively a weighted count)
    1 and 1 is base stirling diversity / raos quadratic
    */
    if class_counts.len() != wt_matrix.len() {
        return Err(exceptions::PyValueError::new_err(
            "Mismatching number of unique class counts and respective class taxonomy tiers.",
        ));
    }
    if wt_matrix[0].len() != wt_matrix.len() {
        return Err(exceptions::PyValueError::new_err(
            "Weights matrix must be an NxN pairwise matrix of disparity weights.",
        ));
    }
    // catch potential division by zero situations
    let num: u32 = class_counts.iter().sum();
    if num < 2 {
        return Ok(0.0);
    }
    // variable for additive calculations of distance * p1 * p2
    let mut raos: f32 = 0.0;
    for (i, &class_count_i) in class_counts.iter().enumerate() {
        for (j, &class_count_j) in class_counts.iter().enumerate() {
            // only need to examine the pair if j > i, otherwise double-counting
            if j > i {
                break;
            }
            let p_i = class_count_i as f32 / num as f32; // place here to catch division by zero for single element
            let p_j = class_count_j as f32 / (num - 1) as f32; // bias adjusted
            let wt = wt_matrix[i][j]; // calculate 3rd level disparity
            raos += wt.powf(alpha) * (p_i * p_j).powf(beta);
        }
    }
    Ok(raos)
}

/*
DEPRECATED in v4

def hill_diversity_pairwise_matrix_wt(
    class_counts: npt.NDArray[np.int_], wt_matrix: npt.NDArray[np.float32], q: np.float32
) -> np.float32:
    """
    Hill diversity weighted by pairwise weights matrix.

    This is the matrix version - requires a precomputed (e.g. disparity) matrix for all classes.

    See above for distance version.

    """
    if len(class_counts) != len(wt_matrix):
        raise ValueError("Mismatching number of unique class counts and dimensionality of class weights matrix.")
    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError("Weights matrix must be an NxN pairwise matrix of disparity weights.")
    if q < 0:
        raise ValueError("Please select a non-zero value for q.")
    # catch potential division by zero situations
    num: int = class_counts.sum()
    if num == 0:
        return np.float32(0)
    # calculate Q
    agg_q = 0
    for i, class_count_i in enumerate(class_counts):
        if not class_count_i:
            continue
        a_i = class_count_i / num
        for j, class_count_j in enumerate(class_counts):
            # only need to examine the pair if j < i, otherwise double-counting
            if j > i:
                break
            if not class_count_j:
                continue
            a_j = class_count_j / num
            wt = wt_matrix[i][j]
            # pairwise distances
            agg_q += wt * a_i * a_j
    # pairwise disparities weights can sometimes give rise to Q = 0... causing division by zero etc.
    if agg_q == 0:
        return np.float32(0)
    # if in the limit, use exponential
    if q == 1:
        div_pw_wt_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i, class_count_i in enumerate(class_counts):
            if not class_count_i:
                continue
            a_i = class_count_i / num
            for j, class_count_j in enumerate(class_counts):
                # only need to examine the pair if j < i, otherwise double-counting
                if j > i:
                    break
                if not class_count_j:
                    continue
                a_j = class_count_j / num
                # pairwise distances
                wt = wt_matrix[i][j]
                div_pw_wt_lim += wt * a_i * a_j / agg_q * np.log(a_i * a_j / agg_q)  # sum
        # once summed
        div_pw_wt_lim = np.exp(-div_pw_wt_lim)
        return np.float32(div_pw_wt_lim ** (1 / 2))  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    div_pw_wt = 0
    for i, class_count_i in enumerate(class_counts):
        if not class_count_i:
            continue
        a_i = class_count_i / num
        for j, class_count_j in enumerate(class_counts):
            # only need to examine the pair if j < i, otherwise double-counting
            if j > i:
                break
            if not class_count_j:
                continue
            a_j = class_count_j / num
            # pairwise distances
            wt = wt_matrix[i][j]
            div_pw_wt += wt * (a_i * a_j / agg_q) ** q  # sum
    div_pw_wt = div_pw_wt ** (1 / (1 - q))
    return np.float32(div_pw_wt ** (1 / 2))  # (FD / Q) ** (1 / 2)
*/

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_network_structure() {
        let counts = vec![1, 3, 1, 1];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let probs = vec![
            0.16666666666666666,
            0.5,
            0.16666666666666666,
            0.16666666666666666,
        ];
        let hd0 = hill_diversity(counts.clone(), 0.0);
        let hd1 = hill_diversity(counts.clone(), 1.0);
        let hd2 = hill_diversity(counts.clone(), 2.0);
        let whd0 =
            hill_diversity_branch_distance_wt(counts.clone(), weights.clone(), 0.0, 0.0, 1.0);
        let whd1 =
            hill_diversity_branch_distance_wt(counts.clone(), weights.clone(), 1.0, 0.0, 1.0);
        let whd2 =
            hill_diversity_branch_distance_wt(counts.clone(), weights.clone(), 2.0, 0.0, 1.0);
        let whpd0 =
            hill_diversity_pairwise_distance_wt(counts.clone(), weights.clone(), 0.0, 0.0, 1.0);
        let whpd1 =
            hill_diversity_pairwise_distance_wt(counts.clone(), weights.clone(), 1.0, 0.0, 1.0);
        let whpd2 =
            hill_diversity_pairwise_distance_wt(counts.clone(), weights.clone(), 2.0, 0.0, 1.0);
        let b = 0;
    }
}

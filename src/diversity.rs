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
    Exponent at 1 = exp(Shannon)
    Exponent at 2 = diversity form of simpson index
    */
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Invalid value for q: must be non-negative.",
        ));
    }
    let num: u32 = class_counts.iter().sum();
    if num == 0 {
        return Ok(0.0);
    }

    if q == 0.0 {
        return Ok(class_counts.iter().filter(|&&count| count > 0).count() as f32);
    }

    let num_f32 = num as f32;

    if (q - 1.0).abs() < f32::EPSILON {
        let entropy: f32 = class_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&class_count| {
                let prob = class_count as f32 / num_f32;
                prob * prob.ln()
            })
            .sum();
        let result = (-entropy).exp();
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Hill diversity calculation resulted in invalid value (q=1 case).",
            ));
        }
        Ok(result)
    } else {
        let diversity_sum: f32 = class_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&class_count| {
                let prob = class_count as f32 / num_f32;
                prob.powf(q)
            })
            .sum();

        if !diversity_sum.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Intermediate Hill diversity sum is invalid.",
            ));
        }

        if diversity_sum < 0.0 && (1.0 / (1.0 - q)).fract() != 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Cannot raise negative base to non-integer exponent in Hill diversity.",
            ));
        }

        let result = diversity_sum.powf(1.0 / (1.0 - q));
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Hill diversity calculation resulted in invalid value.",
            ));
        }
        Ok(result)
    }
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
        return Err(exceptions::PyValueError::new_err(format!(
            "Mismatch between class counts length ({}) and distances length ({}).",
            class_counts.len(),
            class_distances.len()
        )));
    }
    if beta < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Beta must be non-negative.",
        ));
    }
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Invalid value for q: must be non-negative.",
        ));
    }
    let num: u32 = class_counts.iter().sum();
    if num == 0 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;

    let props_and_weights: Vec<(f32, f32)> = class_counts
        .iter()
        .zip(class_distances.iter())
        .filter_map(|(&count, &dist)| {
            if count > 0 {
                let proportion = count as f32 / num_f32;
                match common::clipped_beta_wt(beta, max_curve_wt, dist) {
                    Ok(wt) => Some(Ok((proportion, wt))),
                    Err(e) => Some(Err(e)),
                }
            } else {
                None
            }
        })
        .collect::<PyResult<Vec<(f32, f32)>>>()?;

    let agg_t: f32 = props_and_weights.iter().map(|&(p, w)| w * p).sum();

    if agg_t.abs() < f32::EPSILON {
        return Ok(0.0);
    }

    if (q - 1.0).abs() < f32::EPSILON {
        let weighted_entropy: f32 = props_and_weights
            .iter()
            .filter_map(|&(p, w)| {
                let effective_prop = w * p / agg_t;
                if effective_prop > 0.0 {
                    Some(effective_prop * effective_prop.ln())
                } else {
                    None
                }
            })
            .sum();

        let result = (-weighted_entropy).exp();
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Weighted Hill diversity calculation resulted in invalid value (q=1 case).",
            ));
        }
        Ok(result)
    } else {
        let diversity_sum: f32 = props_and_weights
            .iter()
            .map(|&(p, w)| w * (p / agg_t).powf(q))
            .sum();

        if !diversity_sum.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Intermediate weighted Hill diversity sum is invalid.",
            ));
        }

        if diversity_sum < 0.0 && (1.0 / (1.0 - q)).fract() != 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Cannot raise negative base to non-integer exponent in weighted Hill diversity.",
            ));
        }

        let result = diversity_sum.powf(1.0 / (1.0 - q));
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Weighted Hill diversity calculation resulted in invalid value.",
            ));
        }
        Ok(result)
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
        return Err(exceptions::PyValueError::new_err(format!(
            "Mismatch between class counts length ({}) and distances length ({}).",
            class_counts.len(),
            class_distances.len()
        )));
    }
    if beta < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Beta must be non-negative.",
        ));
    }
    if q < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Invalid value for q: must be non-negative.",
        ));
    }
    let num: u32 = class_counts.iter().sum();
    if num == 0 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;

    // Compute probabilities once for all indices
    let probabilities: Vec<f32> = class_counts
        .iter()
        .map(|&count| count as f32 / num_f32)
        .collect();

    // Use iterators for both i and j for clarity and consistency
    let mut agg_q = 0.0;
    for i in 0..class_counts.len() {
        let count_i = class_counts[i];
        if count_i == 0 {
            continue;
        }
        let a_i = probabilities[i];
        for j in 0..=i {
            let count_j = class_counts[j];
            if count_j == 0 {
                continue;
            }
            let a_j = probabilities[j];
            let wt = common::clipped_beta_wt(
                beta,
                max_curve_wt,
                class_distances[i] + class_distances[j],
            )?;
            agg_q += wt * a_i * a_j;
        }
    }

    if agg_q.abs() < f32::EPSILON {
        return Ok(0.0);
    }

    if (q - 1.0).abs() < f32::EPSILON {
        let mut weighted_entropy_sum = 0.0;
        for i in 0..class_counts.len() {
            let count_i = class_counts[i];
            if count_i == 0 {
                continue;
            }
            let a_i = probabilities[i];
            for j in 0..=i {
                let count_j = class_counts[j];
                if count_j == 0 {
                    continue;
                }
                let a_j = probabilities[j];
                let wt = common::clipped_beta_wt(
                    beta,
                    max_curve_wt,
                    class_distances[i] + class_distances[j],
                )?;
                let effective_prop = wt * a_i * a_j / agg_q;
                if effective_prop > 0.0 {
                    weighted_entropy_sum += effective_prop * effective_prop.ln();
                }
            }
        }
        let exp_val = (-weighted_entropy_sum).exp();
        if !exp_val.is_finite() || exp_val < 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Pairwise Hill diversity intermediate calculation resulted in invalid value (q=1 case).",
            ));
        }
        let result = exp_val.sqrt();
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Pairwise Hill diversity final calculation resulted in invalid value (q=1 case).",
            ));
        }
        Ok(result)
    } else {
        let mut diversity_term_sum = 0.0;
        for i in 0..class_counts.len() {
            let count_i = class_counts[i];
            if count_i == 0 {
                continue;
            }
            let a_i = probabilities[i];
            for j in 0..=i {
                let count_j = class_counts[j];
                if count_j == 0 {
                    continue;
                }
                let a_j = probabilities[j];
                let wt = common::clipped_beta_wt(
                    beta,
                    max_curve_wt,
                    class_distances[i] + class_distances[j],
                )?;
                diversity_term_sum += wt * (a_i * a_j / agg_q).powf(q);
            }
        }

        if !diversity_term_sum.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Intermediate pairwise Hill diversity sum is invalid.",
            ));
        }

        if diversity_term_sum < 0.0 && (1.0 / (1.0 - q)).fract() != 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Cannot raise negative base to non-integer exponent in pairwise Hill diversity.",
            ));
        }
        let pow_val = diversity_term_sum.powf(1.0 / (1.0 - q));
        if !pow_val.is_finite() || pow_val < 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Pairwise Hill diversity intermediate calculation resulted in invalid value (q!=1 case).",
            ));
        }
        let result = pow_val.sqrt();
        if !result.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Pairwise Hill diversity final calculation resulted in invalid value (q!=1 case).",
            ));
        }
        Ok(result)
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
    if num < 2 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;
    let num_minus_1_f32 = (num - 1) as f32;
    let lambda: f32 = class_counts
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| {
            let x_f32 = x as f32;
            (x_f32 / num_f32) * ((x_f32 - 1.0).max(0.0) / num_minus_1_f32)
        })
        .sum();
    Ok((1.0 - lambda).max(0.0))
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
    if num == 0 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;
    let entropy_sum: f32 = class_counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&class_count| {
            let prob = class_count as f32 / num_f32;
            prob * prob.ln()
        })
        .sum();
    if !entropy_sum.is_finite() {
        return Err(exceptions::PyValueError::new_err(
            "Shannon entropy calculation resulted in invalid value.",
        ));
    }
    Ok((-entropy_sum).max(0.0))
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
    let n_classes = class_counts.len();
    if n_classes != wt_matrix.len() {
        return Err(exceptions::PyValueError::new_err(format!(
            "Mismatch between class counts length ({}) and weights matrix rows ({}).",
            n_classes,
            wt_matrix.len()
        )));
    }
    if n_classes > 0 {
        if wt_matrix.is_empty() || wt_matrix[0].len() != n_classes {
            return Err(exceptions::PyValueError::new_err(format!(
                "Weights matrix must be square ({}x{}). Got {}x{}.",
                n_classes,
                n_classes,
                n_classes,
                if wt_matrix.is_empty() {
                    0
                } else {
                    wt_matrix[0].len()
                }
            )));
        }
    }

    if alpha < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Alpha must be non-negative.",
        ));
    }
    if beta < 0.0 {
        return Err(exceptions::PyValueError::new_err(
            "Beta must be non-negative.",
        ));
    }

    let num: u32 = class_counts.iter().sum();
    if num < 2 {
        return Ok(0.0);
    }
    let num_f32 = num as f32;
    let num_minus_1_f32 = (num - 1) as f32;

    let mut raos: f32 = 0.0;
    for i in 0..n_classes {
        let class_count_i = class_counts[i];
        if class_count_i == 0 {
            continue;
        }
        let p_i = class_count_i as f32 / num_f32;

        for j in 0..=i {
            let class_count_j = class_counts[j];
            if class_count_j == 0 {
                continue;
            }

            let p_j = class_count_j as f32 / num_minus_1_f32;
            let wt = wt_matrix[i][j];

            if !wt.is_finite() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid weight encountered in matrix at [{}][{}].",
                    i, j
                )));
            }

            let prob_term = p_i * p_j;
            if !prob_term.is_finite() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid probability product encountered for pair ({}, {}).",
                    i, j
                )));
            }

            let prob_term_pow_beta = if prob_term == 0.0 && beta == 0.0 {
                1.0
            } else {
                if prob_term < 0.0 && beta.fract() != 0.0 {
                    return Err(exceptions::PyValueError::new_err(
                        "Cannot raise negative probability product to non-integer exponent beta.",
                    ));
                }
                prob_term.powf(beta)
            };

            let wt_pow_alpha = if wt == 0.0 && alpha == 0.0 {
                1.0
            } else {
                if wt < 0.0 && alpha.fract() != 0.0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Cannot raise negative weight {} to non-integer exponent alpha {}.",
                        wt, alpha
                    )));
                }
                wt.powf(alpha)
            };

            if !prob_term_pow_beta.is_finite() || !wt_pow_alpha.is_finite() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid intermediate power calculation for pair ({}, {}).",
                    i, j
                )));
            }

            let term = wt_pow_alpha * prob_term_pow_beta;
            raos += term;
        }
    }

    if !raos.is_finite() {
        return Err(exceptions::PyValueError::new_err(
            "Rao's Q calculation resulted in invalid value.",
        ));
    }
    Ok(raos)
}

//
// Created by Jacques Perrault on 10/2/21.
//

#ifndef LAMBDARUNNER_PCA_H
#define LAMBDARUNNER_PCA_H

#include <iostream>
#include <Eigen/Dense>
#include <igl/iterative_closest_point.h>
using namespace Eigen;
using namespace std;

/// perform pca on a matrix of data
/// page 52
/// \param mat n x d matrix of values for pca
/// \return eigenValues, eigenVectors
tuple<VectorXd, MatrixXd> Pca(MatrixXd mat) {
  try {
    // 1. Compute the center of mass of the data points,
    // 2. Translate all the points so that the origin is at m:
    MatrixXd centered = mat.rowwise() - mat.colwise().mean();
    // 3. Construct the d  d scatter matrix S = Y Y >, where Y is the  matrix whose columns are the yi
    MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);
    //4. Compute the spectral decomposition: S = V V >.  5. Sort the eigenvalues in decreasing order: 1  2 
    SelfAdjointEigenSolver<MatrixXd> eigensolver(cov);
    if (eigensolver.info() != Success) throw runtime_error("failed to solve");

    VectorXd eigen_values = eigensolver.eigenvalues();
    MatrixXd eigen_vectors = eigensolver.eigenvectors();

    // 5. sort values and vectors - pretty hacky I wish this was in eigen...
    std::vector<std::tuple<float, VectorXd>> eigen_vectors_and_values;

    for (int i = 0; i < eigen_values.size(); i++) {
      std::tuple<float, VectorXd> vec_and_val(eigen_values[i], eigen_vectors.row(i));
      eigen_vectors_and_values.push_back(vec_and_val);
    }

    std::sort(eigen_vectors_and_values.begin(), eigen_vectors_and_values.end(),
              [&](const std::tuple<float, VectorXd> &a, const std::tuple<float, VectorXd> &b) -> bool {
                return std::get<0>(a) > std::get<0>(b);
              });

    // replace existing matrices with sorted values
    for (int i = 0; i < eigen_vectors_and_values.size(); i++) {
      auto cur = eigen_vectors_and_values[i];
      eigen_values[i] = std::get<0>(cur);
      eigen_vectors.row(i) = std::get<1>(cur);
    }
    tuple<VectorXd, MatrixXd> final(eigen_values, eigen_vectors);
    return final;
  } catch (exception &e) {
    throw e;
  }
}

/// find a rigid transformation between two point sets
/// page 61
/// \param source d x n matrix of values
/// \param target d x n matrix of values
/// \return rotationMatrix, translationVector
tuple<MatrixXd, VectorXd> findRigidTransform(MatrixXd source, MatrixXd target) {
  try {
    //  1. Compute the weighted centroids of both point sets:
    Eigen::RowVectorXd sourceCenter = source.colwise().mean();
    Eigen::RowVectorXd targetCenter = target.colwise().mean();
    //  2. Compute the centered vectors
    //    xi:= pi − p¯, yi:= qi − q¯, i = 1, 2, . . . , n.
    MatrixXd sourceTranslated = source.rowwise() - sourceCenter;
    MatrixXd targetTranslated = target.rowwise() - targetCenter;

    //  3. Compute the d × d covariance matrix
    //  S = XW Y T
    //  where X and Y are the d × n matrices that have xi and yi as their columns, respectively,
    //  and W = diag(w1, w2, . . . , wn).
    auto W = sourceTranslated.transpose() * targetTranslated;

    //  4. Compute the singular value decomposition S = UΣV
    JacobiSVD<Eigen::MatrixXd> svd;
    svd.compute(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (!svd.computeU() || !svd.computeV()) throw runtime_error("failed to compute svd");

    //  5. Compute the optimal translation as
    //  t = q¯ − Rp¯.
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    Eigen::MatrixXd R = U * V.transpose();

    // if the determinate is flipped we need to flip the third column of V
    if (R.determinant() < 0){
      cout << "determinate of svd < 0" << endl;
      V.col(2) *= -1;
      R = V.transpose() * U;
    }
    Eigen::RowVectorXd t = targetCenter.transpose() - R * sourceCenter.transpose();

    return tuple<MatrixXd, VectorXd>(R, t);
  } catch (exception &e) {
    throw e;
  }
}


#endif //LAMBDARUNNER_PCA_H

//
// Created by Jacques Perrault on 12/20/21.
//

#ifndef INC_3DMETHODS_FITTER_TEST_ICP_H_
#define INC_3DMETHODS_FITTER_TEST_ICP_H_

#include <igl/iterative_closest_point.h>
tuple<VectorXd, MatrixXd> icp(MatrixXd mat) {
  // Inline mesh of a cube
  const Eigen::MatrixXd V = (Eigen::MatrixXd(8, 3) <<
                                                   0.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 1.0,
      1.0, 1.0, 0.0,
      1.0, 1.0, 1.0).finished();

  const Eigen::MatrixXi F = (Eigen::MatrixXi(12, 3) <<
                                                    1, 7, 5,
      1, 3, 7,
      1, 4, 3,
      1, 2, 4,
      3, 8, 7,
      3, 4, 8,
      5, 7, 8,
      5, 8, 6,
      1, 5, 6,
      1, 6, 2,
      2, 6, 8,
      2, 8, 4).finished().array() - 1;

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_face_based(true);
  viewer.launch();
}
#endif //INC_3DMETHODS_FITTER_TEST_ICP_H_

/*
 *  SubmodularGreedy.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/SubmodularGreedy.hpp>

namespace NetworKit {
std::ostream &operator<<(std::ostream &os, const Edge &E) {
    os << "Edge(" << E.u << ", " << E.v << ")";
    return os;
}
} // namespace NetworKit
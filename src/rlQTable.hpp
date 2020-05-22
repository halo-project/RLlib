#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <gsl/gsl_vector.h>

namespace rl {
namespace gsl {

/// a table-based parameterization of the q_pi(s,a) estimate for policy pi
template<typename State, int NumStates, typename Action, int NumActions>
class QTable {
  public:
  QTable() {
    size_t Card = NumStates * NumActions;
    assert(Card > 0);
    theta = gsl_vector_alloc(Card);
    gsl_vector_set_zero(theta);
  }

  QTable(const QTable&) = delete;             // not CopyConstructable
  QTable& operator=(QTable const&) = delete;  // not CopyAssignable

  // MoveConstructable is okay, since we won't leak the table pointer.
  QTable(QTable &&T) {
    theta = T.theta; T.theta = nullptr;
  }


  ~QTable() {
    gsl_vector_free(theta);
  }

  /// Creates an independent function object that maintains a copy of the pointer to
  /// the table so that it can perform lookups. Importantly, the destruction of the
  /// function object will _not_ destroy the table.
  std::function<double(State,Action)> curriedQ() {
    return std::bind(Q, theta, std::placeholders::_1, std::placeholders::_2);
  }

  /// provides a pointer to the Q table, but does NOT pass ownership!
  gsl_vector* getTable() { return theta; }

  /// gradient table access, not specific to a particular gradient table
  static void GradQ(const gsl_vector* theta, gsl_vector* grad_theta_sa, State s, Action a) {
    assert(grad_theta_sa && grad_theta_sa->size > 0 && "invalid / freed vector");
    gsl_vector_set_basis(grad_theta_sa, ComputeIndex(s, a));
  }

  /// table access, not specific to a particular table
  static double Q(const gsl_vector* theta, State s, Action a) {
    assert(theta && theta->size > 0 && "invalid / freed vector");
    return gsl_vector_get(theta, ComputeIndex(s, a));
  }

  private:

  static size_t ComputeIndex(State s, Action a) {
    int aInt = static_cast<int>(a);
    assert(0 <= s && s < NumStates && "invalid state");
    assert(0 <= aInt && aInt < NumActions && "invalid action");

    // for now, assuming state and action can be mapped to a table index via a cast to size_t
    return aInt * NumStates + s;
  }

  gsl_vector* theta;
};

} // end namespace gsl
} // end namespace rl
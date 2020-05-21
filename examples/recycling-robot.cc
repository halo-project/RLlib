
#include <iostream>
#include <random>

#include "rl.hpp"

// From Example 3.3 in Sutton & Barto.

namespace act {
enum class Action {
  Wait=0,
  Search=1,
  Recharge=2
};
constexpr size_t Size = 3;

std::string toString(Action a) {
  switch (a) {
    case Action::Wait:
      return "WAIT";
    case Action::Search:
      return "SEARCH";
    case Action::Recharge:
      return "RECHARGE";
  };
  return "???";
}
} // end namespace act



namespace env {
// "observation corresponding to current phase"
// where phase is a term for "state of environment"
// rllib names uses phase instead of state to avoid
// some sort of confusion with the internal RL model.
enum class Environment {
  BatteryLow = 0,
  BatteryHigh = 1
};
constexpr size_t Size = 2;

std::string toString(Environment e) {
  switch (e) {
    case Environment::BatteryLow:
      return "BatteryLow";
    case Environment::BatteryHigh:
      return "BatteryHigh";
  };
  return "???";
}
} // end namespace env


// for the robot. this name is kind of hard-coded.
class Simulator {
  public:
  // required type aliases for rllib
  using reward_type = double;
  using observation_type = env::Environment; // aka state_type
  using action_type = act::Action;

  const observation_type& sense() const {
    return State;
  }

  reward_type reward() const { return LastActionReward; }

  void timeStep(const action_type& Act) {

    // TODO: actually perform the action!

    std::cout << "took action " << act::toString(Act)
              << ", yielding environment" << env::toString(State)
              << ", with reward = " << LastActionReward;
  }

  private:
    observation_type State;
    reward_type LastActionReward;
};



// Definition of Reward, S, A, SA, Transition and TransitionSet.
#include "example-defs-transition.hpp"



///////////////////////////////////////////////
#include <gsl/gsl_vector.h>


// a table-based parameterization of the q_pi(s,a) estimate for policy pi
template<typename State, int NumStates, typename Action, int NumActions>
class QTable {
  public:
  QTable() {
    theta = gsl_vector_alloc(NumStates * NumActions);
    gsl_vector_set_zero(theta);
  }

  ~QTable() {
    gsl_vector_free(theta);
  }

  double operator()(State s, Action a) {
    return Q(theta, s, a);
  }

  gsl_vector* getTable() { return theta; }

  // for now, assuming state and action can be mapped to a table index via a cast to size_t
  #define TABLE_SELECT(s,a)   (static_cast<size_t>(a)*NumStates+static_cast<size_t>(s))

  // NOTE: for some reason theta needs to be given, but is not used.
  static void GradQ(const gsl_vector* theta, gsl_vector* grad_theta_sa, State s, Action a) {
    gsl_vector_set_basis(grad_theta_sa, TABLE_SELECT(s,a));
  }

  static double Q(const gsl_vector* theta, State s, Action a) {
    return gsl_vector_get(theta, TABLE_SELECT(s,a));
  }

  #undef TABLE_SELECT

  private:
  gsl_vector* theta;
};

////////////////////////////////////////////////

#include <random>

// this one uses SARSA (on-policy)
// NOTE: compile with,
//    g++ -I../src -std=c++17 recycling-robot.cc -lgsl
//
int main () {
  std::random_device rd;
  std::mt19937 gen(rd());
  using TableTy = QTable<env::Environment, env::Size, act::Action, act::Size>;
  TableTy Q;

  const double GAMMA = .9;
  const double ALPHA = .05;
  const double EPSILON = .2;

  auto critic = rl::gsl::sarsa<S,A>(Q.getTable(),
          GAMMA,ALPHA,
          TableTy::Q,
          TableTy::GradQ);
}
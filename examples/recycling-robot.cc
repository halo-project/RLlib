
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
const Action Begin = Action::Wait; // the one assigned to index 0

std::string toString(Action a) {
  switch (a) {
    case Action::Wait:
      return "WAIT";
    case Action::Search:
      return "SEARCH";
    case Action::Recharge:
      return "RECHARGE";
  };
  return "unknown#" + std::to_string(static_cast<size_t>(a));
}
} // end namespace act



class Environment {
  public:
    enum {
      size = 2,
      start = 0
    };

    enum {
      BatteryLow = 0,
      BatteryHigh = 1
    };

  using phase_type = int;

  static std::string toString(phase_type e) {
    switch (e) {
      case Environment::BatteryLow:
        return "BatteryLow";
      case Environment::BatteryHigh:
        return "BatteryHigh";
    };
    return "unknown#" + std::to_string(static_cast<size_t>(e));
  }
};


// for the robot. this name is kind of hard-coded.
class Simulator {
  public:
  // required type aliases for rllib
  using reward_type = double;
  using observation_type = Environment::phase_type; // aka state_type
  using action_type = act::Action;

  const observation_type& sense() const {
    return State;
  }

  reward_type reward() const { return LastActionReward; }

  void timeStep(const action_type& Act) {

    // TODO: actually perform the action!

    std::cout << "took action " << act::toString(Act)
              << ", yielding environment " << Environment::toString(State)
              << ", with reward = " << LastActionReward
              << "\n";
  }

  private:
    observation_type State = Environment::start;
    reward_type LastActionReward = 0.0;
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

  double operator()(State s, Action a) const {
    return Q(theta, s, a);
  }

  gsl_vector* getTable() { return theta; }

  // NOTE: for some reason theta needs to be given, but is not used.
  static void GradQ(const gsl_vector* theta, gsl_vector* grad_theta_sa, State s, Action a) {
    assert(grad_theta_sa != nullptr);
    gsl_vector_set_basis(grad_theta_sa, ComputeIndex(s, a));
  }

  static double Q(const gsl_vector* theta, State s, Action a) {
    assert(theta != nullptr);
    std::cerr << "Q size = " << theta->size << "\n";
    return gsl_vector_get(theta, ComputeIndex(s, a));
  }

  private:

  static size_t ComputeIndex(State s, Action a) {
    // for now, assuming state and action can be mapped to a table index via a cast to size_t
    #ifndef NDBUG
      std::cerr << "access of Q("
                << Environment::toString(s) << ", " << act::toString(a) << ")\n";
    #endif

    assert(s < NumStates && "invalid state");
    assert(static_cast<int>(a) < NumActions && "invalid action");

    size_t Index = static_cast<int>(a) * NumStates + s;

    std::cerr << "Index = " << Index << "\n";
    return Index;
  }

  gsl_vector* theta;
};

////////////////////////////////////////////////

#include <random>

template<typename CRITIC,typename Q, typename RANDOM_GENERATOR>
void run_experiment(CRITIC& critic, const Q& q, RANDOM_GENERATOR& gen,
                    const double EPSILON, int NumEpisodes) {
  Simulator Sim;
  int MaxEpisodeSteps = 10;
  double CurrentEpsilon = EPSILON;

  // TODO: these iterators should be part of the Q table!!
  using ActionIteratorTy = rl::enumerator<Simulator::action_type>;
  auto action_begin = ActionIteratorTy(act::Begin);
  auto action_end = action_begin + act::Size;

  // using StateIteratorTy = rl::enumerator<Simulator::observation_type>;
  // auto state_begin  = StateIteratorTy(env::Begin);
  // auto state_end    = state_begin + env::Size;

  auto learning_policy  = rl::policy::epsilon_greedy<Q, ActionIteratorTy, RANDOM_GENERATOR>
                            (q, CurrentEpsilon, action_begin, action_end, gen);

  for (int epi = 0; epi < NumEpisodes; epi++) {
    // Sim.restart();
    rl::episode::learn(Sim, learning_policy, critic, MaxEpisodeSteps);
  }
}

// this one uses SARSA (on-policy)
// NOTE: compile with,
//    g++ -I../src -std=c++17 recycling-robot.cc -lgsl
//
int main () {
  std::random_device rd;
  std::mt19937 gen(rd());
  using TableTy = QTable<Environment::phase_type, Environment::size, act::Action, act::Size>;
  TableTy Q;

  const double GAMMA = .9;
  const double ALPHA = .05;
  const double EPSILON = .2;

  auto critic = rl::gsl::sarsa<S,A>(Q.getTable(),
          GAMMA,ALPHA,
          TableTy::Q,
          TableTy::GradQ);

  run_experiment(critic, Q, gen, EPSILON, 1);
}
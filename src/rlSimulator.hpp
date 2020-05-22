#pragma once

#include <cassert>

namespace rl {

template<typename World, typename Action>
class Simulator {
  public:
  // standard, required members
  using reward_type = double;
  using observation_type = typename World::type; // aka phase_type or state_type
  using action_type = typename Action::type;

  Simulator() = delete;
  // initialize the simulator with an initial reward and state
  Simulator(reward_type r, observation_type s) : InitialState(s), State(s), InitialReward(r), LastActionReward(r) {}

  const observation_type& sense() const {
    return State;
  }

  /// perform the given action synchonrously, such that it that once this
  /// function returns, the simulation was has advanced with a reward for the
  /// action and the new state.
  virtual void timeStep(const action_type& Act) {
    // NOTE: we avoid pure virtual methods to be compatible with builds that do not use RTTI, i.e., -fno-rtti
    assert(false && "timeStep not implemented! You should override it in your subclass!");
  }

  /// resets the simulator to the initial simulator state
  virtual void restart() {
    State = InitialState;
    LastActionReward = InitialReward;
  }

  /// provides the reward of the last action taken
  reward_type reward() const { return LastActionReward; }

  protected:
    const observation_type InitialState;
    observation_type State; // what is returned by sense()

    const reward_type InitialReward;
    reward_type LastActionReward; // what is returned by reward()
};

} // end namespace
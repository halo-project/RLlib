#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <iterator>
#include <gsl/gsl_vector.h>

#include <rl.hpp>

class Action {
public:
    enum Kind {
        Wait=0,
        Search=1,
        Recharge=2
    };
    using type = Kind;
    static constexpr size_t size = 3;
    static const Kind begin = Kind::Wait; // the one assigned to index 0

    static std::string to_string(Kind K) {
        switch (K) {
            case Wait: return "Wait";
            case Search: return "Search";
            case Recharge: return "Recharge";
            default: return "???";
        };
    }
};

class World {
public:
    // standard, required members
    enum Kind {
        HighBattery=0,
        LowBattery=1
    };
    using type = Kind;
    static constexpr type begin = HighBattery; // the one assigned to index 0
    static constexpr type initial = HighBattery; // the starting phase for running a simulation.
    static constexpr int size = 2;

    static std::string to_string(Kind K) {
        switch (K) {
            case HighBattery: return "HighBattery";
            case LowBattery: return "LowBattery";
            default: return "???";
        };
    }
};

// for the robot. this name is kind of hard-coded.
class Simulator {
  public:
  // standard, required members
  using reward_type = double;
  using observation_type = World::type; // aka phase_type or state_type
  using action_type = Action::type;

  const observation_type& sense() const {
    return State;
  }

  reward_type reward() const { return LastActionReward; }

  void timeStep(const action_type& Act) {

    // TODO: actually perform the action!
    LastActionReward = -2.0 * ( 1.0 + static_cast<double>(Act) );
  }

  // I think these members are _not_ required to use the library
  // but are useful for episodic tasks and testing.
  void restart() {
      State = World::initial;
  }

  private:
    observation_type State = World::initial;
    reward_type LastActionReward = 0.0;
};

// Let us define the parameters.
#define paramGAMMA   .99
#define paramALPHA   .05
#define paramEPSILON .7



// use basic SARSA on-policy
int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    // some type defs
    using Reward = Simulator::reward_type;
    using S =  Simulator::observation_type;
    using A = Simulator::action_type;
    using TableTy = rl::gsl::QTable<S, World::size, A, Action::size>;

    // We need to provide iterators for enumerating all the state and action
    // values. This can be done easily from an enumerators.
    auto action_begin = rl::enumerator<A>(Action::begin);
    auto action_end   = action_begin + Action::size;
    // auto state_begin  = rl::enumerator<S>(World::begin);
    // auto state_end    = state_begin + World::size;


    // This is the dynamical system we want to control.
    // Param      param;
    Simulator  simulator;
    TableTy Table;

    gsl_vector* theta = Table.getTable();
    auto q = Table.curriedQ();

    // Let us now define policies, related to q. The learning policy
    // used is an epsilon-greedy one in the following, while we test the
    // learned Q-function with a geedy policy.
    double epsilon       = paramEPSILON;
    auto learning_policy = rl::policy::epsilon_greedy(q,epsilon,action_begin,action_end, gen);
    auto test_policy     = rl::policy::greedy(q,action_begin,action_end);

    // We intend to learn q on-line, by running episodes, and updating a
    // critic from the transition we get during the episodes. Let us use
    // some GSL-based critic for that purpose.
    auto critic = rl::gsl::sarsa<S,A>(theta,
            paramGAMMA,paramALPHA,
            TableTy::Q,
            TableTy::GradQ);

    // We have now all the elements to start experiments.


    // Let us run episodes with the agent that learns the Q-values.
    const int MAX_EPISODE_LENGTH = 10; // was 0 for unbounded until it reaches a terminal state.
    const int MAX_EPISODES = 1000; // was 10000
    std::cout << "Learning " << std::endl
        << std::endl;

    int episode;
    for(episode = 0; episode < MAX_EPISODES; ++episode) {
        simulator.restart();
        auto actual_episode_length = rl::episode::learn(simulator,learning_policy,critic,
                MAX_EPISODE_LENGTH);
        if(episode % 200 == 0)
            std::cout << "episode " << std::setw(5) << episode+1
                << " : length = " << std::setw(5) << actual_episode_length << std::endl;
    }
    std::cout << std::endl;

    // Let us print the parameters. This can be dumped in a file, rather
    // than printed, for saving the learned Q-value function.
    std::cout << "Learned theta : " << std::endl
        << std::endl
        << theta << std::endl
        << std::endl;

    // We can also gather the transitions from an episode into a collection.
    using TransitionTy = rl::Transition<World, Action, Reward>;
    std::vector<TransitionTy> transition_set;
    simulator.restart();
    unsigned int nb_steps = rl::episode::run(simulator,test_policy,
            std::back_inserter(transition_set),
            TransitionTy::make,TransitionTy::make_terminal,
            MAX_EPISODE_LENGTH);
    std::cout << std::endl
        << "Collected transitions :" << std::endl
        << "---------------------" << std::endl
        << nb_steps << " == " << transition_set.size() << std::endl
        << std::endl;
    for(auto& t : transition_set)
        std::cout << t << std::endl;

    return 0;
}


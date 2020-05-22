#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <iterator>
#include <gsl/gsl_vector.h>

#include "QTable.h"
#include "Transition.h"

// This is the rl header
#include <rl.hpp>

namespace act {
    enum class Action {
        Wait=0,
        Search=1,
        Recharge=2
    };
    constexpr size_t size = 3;
    const Action begin = Action::Wait; // the one assigned to index 0
} // end namespace act

namespace std {
    std::string to_string(act::Action a) {
        return "TODO2";
    }
}

// These are useful typedefs
class World {
public:
    // standard, required members
    using phase_type = int;
    static constexpr phase_type start = 0;
    static constexpr int size = 5;
};

// for the robot. this name is kind of hard-coded.
class Simulator {
  public:
  // standard, required members
  using reward_type = double;
  using observation_type = World::phase_type; // aka state_type
  using action_type = act::Action;

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
      State = World::start;
  }

  private:
    observation_type State = World::start;
    reward_type LastActionReward = 0.0;
};

typedef Simulator::reward_type                               Reward;
typedef Simulator::observation_type                          S;
typedef Simulator::action_type                               A;

// Let us define the parameters.
#define paramGAMMA   .99
#define paramALPHA   .05
#define paramEPSILON .7



// use basic SARSA on-policy
int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    // We need to provide iterators for enumerating all the state and action
    // values. This can be done easily from an enumerators.
    auto action_begin = rl::enumerator<A>(act::begin);
    auto action_end   = action_begin + act::size;
    auto state_begin  = rl::enumerator<S>(World::start);
    auto state_end    = state_begin + World::size;


    // This is the dynamical system we want to control.
    // Param      param;
    Simulator  simulator;
    using TableTy = QTable<S, World::size, A, act::size>;
    TableTy Table;

    // Our Q-function is determined by some vector parameter. It is a
    // gsl_vector since we use the GSL-based algorithm provided by the
    // library.
    // gsl_vector* theta = gsl_vector_alloc(TABULAR_Q_CARDINALITY);
    // gsl_vector_set_zero(theta);
    gsl_vector* theta = Table.getTable();

    // If we need to use the Q-function parametrized by theta as q(s,a),
    // we only have to bind our q_from_table function and get a
    // functional object.
    auto q = Table.curriedQ();

    // Let us now define policies, related to q. The learning policy
    // used is an epsilon-greedy one in the following, while we test the
    // learned Q-function with a geedy policy.
    double epsilon       = paramEPSILON;
    auto learning_policy = rl::policy::epsilon_greedy(q,epsilon,action_begin,action_end, gen);
    auto test_policy     = rl::policy::greedy(q,action_begin,action_end);

    // We intend to learn q on-line, by running episodes, and updating a
    // critic fro the transition we get during the episodes. Let us use
    // some GSL-based critic for that purpose.
    auto critic = rl::gsl::sarsa<S,A>(theta,
            paramGAMMA,paramALPHA,
            TableTy::Q,
            TableTy::GradQ);

    // We have now all the elements to start experiments.


    // Let us run 10000 episodes with the agent that learns the Q-values.
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


    // Let us define v as v(s) = max_a q(s_a) with a labda function.
    // auto v = [&action_begin,&action_end,&q](S s) -> double {return rl::max(std::bind(q,s,_1),
    //         action_begin,
    //         action_end);};
    // // We can draw the Value function a image file.
    // auto v_range = rl::range(v,state_begin,state_end);
    // std::cout << std::endl
    //     << " V in [" << v_range.first << ',' << v_range.second << "]." << std::endl
    //     << std::endl;
    // // World::draw("V-overview",0,v,v_range.first,v_range.second);
    // std::cout << "Image file \"V-overview-000000.ppm\" generated." << std::endl
    //     << std::endl;


    // Let us be greedy on the policy we have found, using the greedy
    // agent to run an episode.
    // simulator.restart();
    // unsigned int nb_steps = rl::episode::run(simulator,test_policy,MAX_EPISODE_LENGTH);
    // std::cout << "Best policy episode ended after " << nb_steps << " steps." << std::endl;

    // We can also gather the transitions from an episode into a collection.
    using TransitionTy = Transition<S, A, Reward>;
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


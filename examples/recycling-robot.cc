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


class RobotSimulator : public rl::Simulator<World, Action> {
public:
    RobotSimulator() : rl::Simulator<World, Action>(0.0, World::initial), rd(), gen(rd()) {}

    void transition(observation_type s, reward_type r) {
        State = s;
        LastActionReward = r;
    }

    /// @returns true with the given probability in [0, 1]
    bool trueWithProbability(double probabilityTrue) {
        std::uniform_real_distribution<double> dist(0, 1); // returns [a, b)
        return dist(gen) < probabilityTrue;
    }

    void timeStep(const action_type &a) override {
        if (State == World::HighBattery) {

            if (a == Action::Search)
                return trueWithProbability(alpha)
                        ? transition(World::HighBattery, RewardSearch)
                        : transition(World::LowBattery, RewardSearch);

            if (a == Action::Wait)
                return trueWithProbability(1.0)
                        ? transition(World::HighBattery, RewardWait)
                        : transition(World::LowBattery, RewardWait);

            if (a == Action::Recharge)
                // TODO: should we give a penalty or return "bad action" for this case?? will the runner understand?
                return transition(World::HighBattery, RewardBatteryDie);

        } else if (State == World::LowBattery) {

            if (a == Action::Search)
                return trueWithProbability(beta)
                        ? transition(World::LowBattery, RewardSearch)
                        : transition(World::HighBattery, RewardBatteryDie);

            if (a == Action::Wait)
                return trueWithProbability(1.0)
                        ? transition(World::LowBattery, RewardWait)
                        : transition(World::HighBattery, RewardWait);

            if (a == Action::Recharge)
                return trueWithProbability(1.0)
                        ? transition(World::HighBattery, RewardRecharge)
                        : transition(World::LowBattery, RewardRecharge);

        }

        assert(false && "bad state or action");
    }

private:

    std::random_device rd;
    std::mt19937 gen;

    // probabilities to adjust the model
    double alpha = .95; // probabilty of battery remaining high while searching in high state
    double beta = .45; // probability of battery remaining low while searching in low state

    reward_type RewardSearch = 1.; // r_search
    reward_type RewardWait = .0; // r_wait
    reward_type RewardBatteryDie = -2.;
    reward_type RewardRecharge = .0;
};

// Let us define the parameters.
#define paramGAMMA   .99    // discount rate
#define paramALPHA   .05    // step size
#define paramEPSILON .7     // how greedy we should be in selection actions



// use basic SARSA on-policy
int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    // some type defs
    using Reward = RobotSimulator::reward_type;
    using S =  RobotSimulator::observation_type;
    using A = RobotSimulator::action_type;
    using TableTy = rl::gsl::QTable<S, World::size, A, Action::size>;

    // We need to provide iterators for enumerating all the state and action
    // values. This can be done easily from an enumerators.
    auto action_begin = rl::enumerator<A>(Action::begin);
    auto action_end   = action_begin + Action::size;
    // auto state_begin  = rl::enumerator<S>(World::begin);
    // auto state_end    = state_begin + World::size;


    // This is the dynamical system we want to control.
    // Param      param;
    RobotSimulator  simulator;
    TableTy Table;

    gsl_vector* theta = Table.getTable();
    auto q = Table.curriedQ();

    // Let us now define policies, related to q. The learning policy
    // used is an epsilon-greedy one in the following, while we test the
    // learned Q-function with a geedy policy.
    double epsilon       = paramEPSILON;
    auto learning_policy = rl::policy::epsilon_greedy(q,epsilon,action_begin,action_end, gen);

    // auto test_policy     = rl::policy::greedy(q,action_begin,action_end);
    double test_epsilon = 0.9;
    auto test_policy = rl::policy::epsilon_greedy(q,test_epsilon,action_begin,action_end, gen);


    // We intend to learn q on-line, by running episodes, and updating a
    // critic from the transition we get during the episodes. Let us use
    // some GSL-based critic for that purpose.
    auto critic = rl::gsl::sarsa<S,A>(theta,
            paramGAMMA,paramALPHA,
            TableTy::Q,
            TableTy::GradQ);

    // We have now all the elements to start experiments.


    // Let us run episodes with the agent that learns the Q-values.
    const int MAX_EPISODE_LENGTH = 1000; // was 0 for unbounded until it reaches a terminal state.
    const int MAX_EPISODES = 20000; // was 10000
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


#pragma once

/// A representation of a state transition.
template<typename State, typename Action, typename Reward>
struct Transition {
    State   s;
    Action  a;
    Reward  r;
    State   s_; // read s_ as s'
    bool   is_terminal;

  static Transition make(State s, Action a, Reward r, State s_) {
    return {s,a,r,s_,false};
  }

  static Transition make_terminal(State s, Action a, Reward r) {
    return {s,a,r,s /* unused */,true};
  }
};

// This prints a transition.
template<typename State, typename Action, typename Reward>
std::ostream& operator<<(std::ostream& os, const Transition<State, Action, Reward>& t) {
    os << std::setw(3) << t.s  << ' ' <<"<ACTION NAME>" // std::to_string(t.a)
        << " ---" << std::setw(5) << t.r << " ---> ";
    if(t.is_terminal)
        os << "End-of-Episode";
    else
        os << std::setw(3) << t.s_;
    return os;
}
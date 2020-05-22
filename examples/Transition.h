#pragma once

/// A representation of a state transition.
template<typename State, typename Action, typename Reward>
struct Transition {
    using ActionKind = typename Action::type;

    State         s;
    ActionKind    a;
    Reward        r;
    State         s_; // read s_ as s'
    bool is_terminal;

  static Transition make(State s, ActionKind a, Reward r, State s_) {
    return {s,a,r,s_,false};
  }

  static Transition make_terminal(State s, ActionKind a, Reward r) {
    return {s,a,r,s /* unused */,true};
  }
};

template<typename State, typename Action, typename Reward>
std::ostream& operator<<(std::ostream& os, const Transition<State, Action, Reward>& t) {
    os << std::setw(3) << t.s  << ' ' << Action::to_string(t.a) // std::to_string(t.a)
        << " ---" << std::setw(5) << t.r << " ---> ";
    if(t.is_terminal)
        os << "End-of-Episode";
    else
        os << std::setw(3) << t.s_;
    return os;
}
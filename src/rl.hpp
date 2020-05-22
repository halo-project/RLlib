/*   This file is part of rl-lib
 *
 *   Copyright (C) 2010,  Supelec
 *
 *   Author : Herve Frezza-Buet and Matthieu Geist
 *
 *   Contributor :
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU General Public
 *   License (GPL) as published by the Free Software Foundation; either
 *   version 3 of the License, or any later version.
 *
 *   This library is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *   General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 *   Contact : Herve.Frezza-Buet@supelec.fr Matthieu.Geist@supelec.fr
 *
 */

#pragma once

#include <gsl/gsl_vector.h>
#include <cmath>

#include <rlAlgo.hpp>
#include <rlEpisode.hpp>
#include <rlException.hpp>
#include <rlKTD.hpp>
#include <rlLSTD.hpp>
#include <rlMLP.hpp>
#include <rlOffPAPI.hpp>
#include <rlPolicy.hpp>
#include <rlQLearning.hpp>
#include <rlSARSA.hpp>
#include <rlTD.hpp>
#include <rlActorCritic.hpp>
#include <rlTypes.hpp>

/////
//  example problems
// #include <rl-boyan-chain.hpp>
// #include <rl-mountain-car.hpp>
// #include <rl-inverted-pendulum.hpp>
// #include <rl-cliff-walking.hpp>
// #include <rl-garnet.hpp>




/**
 * @example example-000-000-overview.cc
 */

/**
 * @example example-000-001-simulator.cc
 */

/**
 * @example example-000-002-learning.cc
 */

/**
 * @example example-000-003-agents.cc
 */

/**
 * @example example-001-001-cliff-walking-sarsa.cc
 */

/**
 * @example example-001-002-cliff-walking-qlearning.cc
 */


/**
 * @example example-002-001-boyan-lstd.cc
 */

/**
 * @example example-002-002-pendulum-lspi.cc
 */

/**
 * @example example-003-001-pendulum-ktdq.cc
 */

/**
 * @example example-003-002-pendulum-mlp-ktdq.cc
 */

/**
 * @example example-003-003-mountain-car-ktdsarsa.cc
 */

/**
 * @example example-004-001-cliff-onestep.cc
 */

/**
 * @example example-004-002-cliff-eligibility.cc
 */

/**
 * @example example-defs-transition.hpp
 */

/**
 * @example example-defs-tabular-cliff.hpp
 */

/**
 * @example example-defs-cliff-experiments.hpp
 */

/**
 * @example example-defs-pendulum-architecture.hpp
 */

/**
 * @example example-defs-test-iteration.hpp
 */

/**
 * @example example-defs-ktdq-experiments.hpp
 */

/**
 * @example example-defs-mountain-car-architecture.hpp
 */








/**
 * @mainpage
 *
 * @section Overview
 *
 * The rl library is not a framework where you can plug your own
 * algorithms by complying to predefined interfaces. It is rather a set
 * of tools that helps you designing your work or experiment from
 * scratch. The main function is yours, and you are responsible for
 * scheduling everything from it, for creating every object that you
 * need.
 *
 * In such a design, the library offers ready-to-use algorithms and
 * types written as templates. Using the template is equivalent as
 * asking some programmer to write for you a code that is dedicated to
 * your application.
 *
 * This documentation contains both a reference and a user manual. The
 * reference manual is given by the Doxygen structure of class
 * names, as usually. The user manual is a better way to get familiar
 * with the library. The user manual, here, consists of the set of
 * examples. You can start by reading them, <b>in the suggested
 * order</b> (examples all have a number).
 *
 * The use of templates may be considered as adding programming
 * complexity. The point is that this kind of genericity, based on a
 * re-writing mechanism at compiling time that writes code for you,
 * makes the design close to the mathematics. The cost is that you
 * spend time to make sure that you fit the requirements when you use
 * some rl object. If you don't, you will get some very complicated
 * syntax error message. This is clearly the drawback of the use of
 * templates. Nevertheless, once it compiles, the code you get is
 * quite safe. Our philosophy is that fixing syntax error is a finite
 * process, as opposed to bug fixing.
 *
 * @section tailor Tailoring your code with typedefs
 *
 * As you will see when browsing examples, it often contains a list of
 * typedefs. This is a smart way to cope with quite complicated types
 * generated by templates. For example:
 *
 * @code
typedef rl::problem::mountain_car::DefaultParam        mcParam;
typedef rl::problem::mountain_car::Simulator<mcParam>  Simulator;
typedef Simulator::action_type                         A;
 * @endcode
 * So when you write afterward:
 * @code
 A optimal_action;
 * @endcode
 * It is as if you had written:
 * @code
 rl::problem::mountain_car::Simulator<rl::problem::mountain_car::DefaultParam>::action_type optimal_action;
 * @endcode
 * This raises a problem when syntax error occur, since error message displays the complicated version of your types.
 *
 * @section concept The use of concepts
 *
 * There is not a clean support of concepts in the version of c++ that
 * we use. In order to help the designers, we have made explicit concepts
 * in classes, that are just aimed at being documented here, and that
 * are never used in the code. They are gathered in the rl::concept
 * namespace. The convention is the following. If you need a rl
 * template whose documentation is like this:
 *
 * @code
namespace rl {
  template<typename STUFF, typename SA_FOO, typename SA_BAR>
  class DummyAlgorithm {
  public:
    double computeResult(void);
  };
}
 * @endcode
 *
 * You have to search in the documentation for concepts
 * rl::concept::Stuff, rl::concept::sa::Foo, rl::concept::sa::Bar, as
 * suggests the names of the formal template parameters of
 * DummyAlgorithm. Let us suppose that you find this.
 *
 * @code
namespace rl {
  namespace concept {

    template<typename ANY>
    class Stuff {
    public:
      typedef ANY any_type;
      void interpret(any_type& a);
    };

    namespace sa {
      template<typename VALUE>
      class FooBase {
      public:
        typedef VALUE value_type;
        value_type get(void);
      };

      template<typename VALUE>
      class Foo : public FooBase {
      public:
        void set(const value_type& v);
      };

      class Bar {
      public:
         static int size(void);
      };
    }
  }
}
 * @endcode
 *
 * It does not mean at all that you have to inherit from the previous
 * classes in order to provide type parameters to the DummyAlgorithm
 * class. Rather, it means that you have to design a class <b>accordingly</b>
 * to the concept classes. Let us make a class that fits all the three
 * rl::concept::Stuff, rl::concept::sa::Foo and rl::concept::sa::Bar
 * concepts. You just have to copy-paste from the concept
 * documentation.
 *
 * @code
class ThreeInOne {
public:

  // This fits rl::concept::Stuff<std::string>

  typedef std::string any_type;

  void interpret(any_type& a) {
    // your code here
  }

  // This fits rl::concept::sa::Foo<int>... and
  // rl::concept::sa::FooBase<int> since Foo inherits
  // from FooBase

  typedef int value_type;

  value_type get(void) {
    // your code here
  }

  void set(const value_type& v) {
    // your code here
  }

  // This fits rl::concept::sa::Bar

  static int size(void) {
    // your code here
  }
};
 * @endcode
 *
 * Once this ThreeInOne class is defined, it can be used as a type
 * parameter for the three slots in the DummyAlgorithm template, since
 * it fits the three requirements.
 *
 * @code
typedef rl::DummyAlgorithm<ThreeInOne,ThreeInOne,ThreeInOne> MyAlgo;
...
MyAlgo algo;
double res = algo.computeResult;
 * @endcode
 *
 * Fitting to the concepts ensures that your code will compile. It
 * also induce a very strong type checking, that may be annoying at
 * compiling time if you do not perfectly fit the concepts, but that
 * brings a lot of safety at run time.
 *
 *
 * @section functional A intensive use of C++-11 function tools
 *
 * There are quite a few concepts in the library (since version
 * 3.00.00 !). They are used mainly for the definition of
 * simulators. Fitting a concept often requires to define wrapper
 * classes, in order to make pr-existing code elements compatible with
 * the required concepts. This is why the rl design is rather based on
 * functions, as examples show. Lambda functions and bindings are
 * widely used in the examples, since they provide a powerfull and
 * compact way to wrap things.
 *
 * This is an example of the use of bindings.
 * @code
double q_param(const Param& theta, S s, A a) {
  // compute q_\theta(s,a)
}

using namespace std::placeholders; // defines _1,_2,...

Param p;

// A Q-function takes two arguments, not three. We can get a
// Q-function by binding the first parameter of q_param to p.

auto q1 = std::bind(q_param,p,_1,_2); // q1(x,y) = q_param(p,x,y).

// The same can be done from a lambda function as well.

auto q2 = [&p](S s, A a) -> double {return q_param(p,s,a);};

S s0;

// If we want to get the best action from s0, we need an action
// iterator (got from an array here) and a binding.

std::array<A,NB_ACTIONS> actions = {{action1,action2,action3,...}};

auto a_q_pair = rl::argmax(std::bind(q1,s0,_1), // this is f(x) = q1(s0,x).
                           actions.begin(), actions.end());
auto best_a   = a_q_pair.first;
 * @endcode
 *
 * Another use of functions is to provide accessors to internal data,
 * or builders, so that algorithm can handle a data without requiring
 * some template fitting. This avoids the above mentionned concept-based
 * wrapping. Let us see an example that gives the taste of the rl
 * library design.
 *
 * Let us suppose that the library provides an algorithm that sums the
 * real parts of a collection of complex numbers. The sumation
 * algotithm would be written like this.
 *
 * @code
template<typename ITERATOR, typename GET_COMPLEX, typename GET_REAL>
double sum_real_parts(const ITERATOR& begin, const ITERATOR& end,
                      const GET_COMPLEX& get_complex,
                      const GET_COMPLEX& get_real) {
  double sum = 0;
  for(auto it = begin; it != end; ++it)
    sum += get_real(get_complex(*it));
  return sum;
}
 * @endcode
 *
 * The previous code do not expect that the complex are placed within
 * a vector, since general purpose iterators are expected. Moreover,
 * the content of the collection has not to be directly a complex,
 * since get_complex is invoqued to get it. Last, complex are note
 * required to fit some concept telling that c.re has to be a legal
 * expression, since get_real does the job. Let use our algorithm with
 * some data.
@code
typedef std::pair<double,double> Complex;
struct Data {
  Complex     value;
  std::string name;
  int         tag;
};
std::map<std::string,Data> database = .... ;

double sum = sum_real_parts(database.begin(),database.end(),
			    [](const Data& content) -> Complex {return content.second.value;}, // Gets the complex...
			    [](const Complex& c)    -> double  {return c.first;});             // ... and from it, its real part.
 * @endcode
 *
 * @section start Getting started
 *
 * You are now ready to read the examples following the order induced
 * by the file names, and of course you can design you own
 * experiments, inspiring from the code in the examples. In order to
 * compile your code, pkg-config support is available (unix).
 *
 * @code
g++ -o example.bin file.cc `pkg-config --cflags --libs rl`
./example.bin
 * @endcode
 * or more generally
 * @code
g++ -c file1.cc `pkg-config --cflags rl`
g++ -c file2.cc `pkg-config --cflags rl`
...
g++ -c fileN.cc `pkg-config --cflags rl`

g++ -o example.bin *.o `pkg-config --libs rl`
./example.bin
 * @endcode
 */



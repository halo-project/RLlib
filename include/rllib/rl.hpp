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

#include "../../src/rlAlgo.hpp"
#include "../../src/rlEpisode.hpp"
#include "../../src/rlException.hpp"
#include "../../src/rlKTD.hpp"
#include "../../src/rlLSTD.hpp"
#include "../../src/rlMLP.hpp"
#include "../../src/rlOffPAPI.hpp"
#include "../../src/rlPolicy.hpp"
#include "../../src/rlQLearning.hpp"
#include "../../src/rlQTable.hpp"
#include "../../src/rlSARSA.hpp"
#include "../../src/rlSimulator.hpp"
#include "../../src/rlTD.hpp"
#include "../../src/rlActorCritic.hpp"
#include "../../src/rlTransition.hpp"
#include "../../src/rlTypes.hpp"

/*
 * Copyright (C) 2016 IIT-ADVR
 * Author: Arturo Laurenzi, Luca Muratore
 * email:  arturo.laurenzi@iit.it, luca.muratore@iit.it
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>
*/

#ifndef __FILTER_UTILS__
#define __FILTER_UTILS__

#include <memory>

namespace tpo { namespace utils {
    
template <typename SignalType>
/**
 * @brief SecondOrderFilter implements a canonical continuous-time
 * second order filter with transfer function
 *                   1
 * P(s) =  -----------------------,  w = natural frequency, eps = damping ratio
 *         (s/w)^2 + 2*eps/w*s + 1
 *
 * and discretized according to a trapezoidal (aka Tustin) scheme. This yields
 * a difference equation of the following form:
 *
 *      a0*y + a1*yd + a2*ydd = u + b1*ud + b2*udd
 *
 * where yd = y(k-1), ydd = y(k-2) and so on (d = delayed).
 */
class SecondOrderFilter {

public:

    typedef std::shared_ptr<SecondOrderFilter<SignalType>> Ptr;

    SecondOrderFilter():
        _omega(1.0),
        _eps(0.8),
        _ts(0.01),
        _reset_has_been_called(false)
    {
        computeCoeff();
    }

    SecondOrderFilter(double omega, double eps, double ts, const SignalType& initial_state):
        _omega(omega),
        _eps(eps),
        _ts(ts),
        _reset_has_been_called(false)
    {
        computeCoeff();
        reset(initial_state);
    }

    void reset(const SignalType& initial_state){
        _reset_has_been_called = true;
        _u = initial_state;
        _y = initial_state;
        _yd = initial_state;
        _ydd = initial_state;
        _udd = initial_state;
        _ud = initial_state;
    }

    const SignalType& process(const SignalType& input){

        if(!_reset_has_been_called) reset(input*0);


        _ydd = _yd;
        _yd = _y;
        _udd = _ud;
        _ud = _u;


        _u = input;
        _y = 1.0/_a0 * ( _u + _b1*_ud + _b2*_udd - _a1*_yd - _a2*_ydd );

        return _y;
    }

    const SignalType& getOutput() const {
        return _y;
    }

    void setOmega(double omega){
        _omega = omega;
        computeCoeff();
    }

    double getOmega()
    {
        return _omega;
    }

    void setDamping(double eps){
        _eps = eps;
        computeCoeff();
    }

    double getDamping()
    {
        return _eps;
    }

    void setTimeStep(double ts){
        _ts = ts;
        computeCoeff();
    }

    double getTimeStep()
    {
        return _ts;
    }

private:

    void computeCoeff()
    {
        _b1 = 2.0;
        _b2 = 1.0;

        _a0 = 1.0 + 4.0*_eps/(_omega*_ts) + 4.0/std::pow(_omega*_ts, 2.0);
        _a1 = 2 - 8.0/std::pow(_omega*_ts, 2.0);
        _a2 = 1.0 + 4.0/std::pow(_omega*_ts, 2.0) - 4.0*_eps/(_omega*_ts);

    }

    double _omega;
    double _eps;
    double _ts;

    double _b1, _b2;
    double _a0, _a1, _a2;

    bool _reset_has_been_called;

    SignalType _y, _yd, _ydd, _u, _ud, _udd;

};

template <typename SignalType>
class FilterWrap {
    
public:
    
    typedef std::shared_ptr<FilterWrap<SignalType>> Ptr;
    
    FilterWrap(const double& DAMPING_FACT, const double& BW_MEDIUM, const double& periodSec, const unsigned& dim) :
        _dim(dim)
    {

        double omega = 2.0 * M_PI * BW_MEDIUM;

        _filter.setDamping ( DAMPING_FACT );
        _filter.setTimeStep ( periodSec );
        _filter.setOmega ( omega );

        reset();
        
    };
    
    void reset(const double& DAMPING_FACT, const double& BW_MEDIUM) {
        
        double omega = 2.0 * M_PI * BW_MEDIUM;

        _filter.setDamping ( DAMPING_FACT );
        _filter.setOmega ( omega );
        
        reset();
    }
    
    void reset() {
        Eigen::VectorXd resetVect = Eigen::VectorXd::Zero(_dim);
        _filter.reset (resetVect);     
    }
    
    const SignalType& process(const SignalType& input) {
        return (_filter.process(input));
    }

private:    
    SecondOrderFilter<SignalType> _filter;
    const unsigned int _dim;
};

}} 

#endif // __FILTER_UTILS__

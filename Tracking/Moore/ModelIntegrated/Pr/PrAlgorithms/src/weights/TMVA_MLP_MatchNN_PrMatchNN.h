/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "Kernel/STLExtensions.h"
#include "Kernel/TMV_utils.h"
#include "LHCbMath/SIMDWrapper.h"
#include <array>
#include <string_view>

/**
 * @file TMVA_MLP_MatchNN_PrMatchNN.h
 * @brief The weights in this file are produced by the Reco-Parameterisation-Tuner
 * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner
 */

using simd = SIMDWrapper::best::types;

namespace LHCb::Pr::MatchNN {
  namespace NN {

    auto ActivationFnc       = []( simd::float_v x ) { return select( x > 0, x, 0 ); };
    auto OutputActivationFnc = []( simd::float_v x ) {
      // sigmoid
      return 1.f / ( 1.f + exp( -x ) );
    };

    const auto fMin =
        std::array<simd::float_v, 6>{ { 6.2234539655e-06, 1.07554035367e-06, 0, 0, 1.38022005558e-06, 0 } };
    const auto fMax = std::array<simd::float_v, 6>{
        { 14.9999675751, 0.414966464043, 249.946044922, 399.411682129, 1.32134592533, 0.148659110069 } };
    const auto fWeightMatrix0to1 = std::array<std::array<simd::float_v, 7>, 8>{
        { { -1.81318680192985, 11.5306183035191, -1.52244588205196, -2.18285669265567, 5.01352644485465,
            -5.51296033910149, 5.73927468893956 },
          { -0.672534709795381, -3.00002957605882, 6.88356805276872, -6.22160659721202, 6.77446979297102,
            3.22745998562836, 2.16560576533548 },
          { 0.671467962865227, -5.25794414846222, 19.3828230421486, 11.0803546893003, -6.38234816567783,
            -8.90286557784295, 10.7684525390767 },
          { -0.2692056487945, -45.0124720478328, 3.02956760827695, -5.39985352881923, 2.33235637852444,
            3.67377088731803, -41.6892338123688 },
          { -1.7097866252219, -2.44815463022872, -6.25060061923427, -2.9527155271918, -2.82646287573035,
            -2.57930159017213, -15.3820440704287 },
          { -1.05477315994645, 10.922735030486, 3.15543979640938, -1.83775727341147, 7.65261550754585,
            -6.94317448033313, 6.86131922732798 },
          { -0.79066972900182, -0.617757099680603, 0.740878002718091, 0.681870030239224, -1.20759406685829,
            0.769290467724204, -1.8437808630988 },
          { -0.184133955272691, 1.92932229057759, 10.2040343486098, 4.08783185462586, -2.02695228923391,
            -3.00792235466827, 10.2821397360227 } } };
    const auto fWeightMatrix1to2 = std::array<std::array<simd::float_v, 9>, 6>{
        { { -0.529669554811976, -2.45282233466048, 1.45989990967879, 3.56480948423982, 0.687553026936273,
            1.78027012856298, 1.63438201788813, -2.94255147008571, -2.10797233521637 },
          { 1.36475059953963, 0.542190986793164, -0.135276688209357, -0.761685823733301, 0.679401991574712,
            -1.40198671179551, -1.61531096417457, -0.791464040720268, 0.852677079400607 },
          { 0.767942415115046, -2.97714597002192, -3.5629451506092, -2.69040161409325, 3.21229316674369,
            0.688654835034672, -0.825543426908553, -1.84996857815595, -7.69537697905136 },
          { 0.114639040310829, -0.37219550277267, -1.42908394861416, -1.86752756108709, -0.839837159377482,
            -1.70735346337309, 1.61348068527877, -1.66550797875971, -0.949665027488677 },
          { -0.0439008856537062, 0.14714685191285, -0.900218617709006, 0.734110875341394, -3.26381964641836,
            -0.903556360012639, -0.848898627795279, 2.4264150318668, 0.290359165274663 },
          { 0.404515384352441, -0.158287682443141, -1.5660040193724, -1.64457334373498, 0.883554107720622,
            -1.48730815915072, -1.52203810494393, 3.67527716420631, -0.393484682839 } } };
    const auto fWeightMatrix2to3 =
        std::array<simd::float_v, 7>{ { -0.776910463978178, 0.811895970822024, 0.775804138783722, 0.282335113136984,
                                        -0.612856158181358, 0.786801771324536, -2.16123706007375 } };

    // Normalization transformation
    const auto transformer = TMV::Utils::Transformer{ fMin, fMax };

    // the training input variables
    const auto validator = TMV::Utils::Validator{
        "ReadMLPMatching", std::tuple{ "chi2", "teta2", "distX", "distY", "dSlope", "dSlopeY" } };
    const auto l0To1 = TMV::Utils::Layer{ fWeightMatrix0to1, ActivationFnc };
    const auto l1To2 = TMV::Utils::Layer{ fWeightMatrix1to2, ActivationFnc };
    const auto l2To3 = TMV::Utils::Layer{ fWeightMatrix2to3, OutputActivationFnc };
    const auto MVA   = TMV::Utils::MVA{ validator, transformer, 0, l0To1, l1To2, l2To3 };

  } // namespace NN

  struct ReadMLPMatching final {

    // constructor
    ReadMLPMatching( LHCb::span<const std::string_view, 6> theInputVars ) { NN::MVA.validate( theInputVars ); }

    // the classifier response
    // "inputValues" is a vector of input values in the same order as the
    // variables given to the constructor
    static auto GetMvaValue( LHCb::span<const simd::float_v, 6> input ) { return NN::MVA( input ); }
  };

} // namespace LHCb::Pr::MatchNN

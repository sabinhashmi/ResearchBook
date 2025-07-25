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
#include "vdt/exp.h"
#include <array>
#include <string_view>

/**
 * @file TMVA_MLP_GhostNN_PrForwardTracking.h
 * @brief The weights in this file are produced by the Reco-Parameterisation-Tuner
 * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner
 */

namespace LHCb::Pr::Forward {
  namespace NNVeloUT {
    // ReLU
    constexpr auto ActivationFnc       = []( float x ) { return x > 0 ? x : 0; };
    constexpr auto OutputActivationFnc = []( float x ) {
      // sigmoid
      return 1.f / ( 1.f + vdt::fast_expf( -x ) );
    };

    constexpr auto fMin = std::array<float, 10>{
        { -5.79507064819, 0.000539337401278, 6.44962929073e-05, 9.1552734375e-05, 0, 0, 0, 1.27572263864e-09, 0, 0 } };
    constexpr auto fMax =
        std::array<float, 10>{ { 21.3727073669, 7.9990901947, 139.93737793, 499.828094482, 139.837768555,
                                 0.0549617446959, 0.281096696854, 0.00079078116687, 0.349586248398, 0.0576412677765 } };
    constexpr auto fWeightMatrix0to1 = std::array<std::array<float, 11>, 12>{
        { { -5.19794815568925, 1.87197537853549, -1.10717797757926, -2.86970252748238, -6.34055081230473,
            -5.00759687179371, 0.31193986693587, -8.01387747621716, -0.120803639675188, -22.2071141316253,
            -38.8490339403707 },
          { 20.4100238181115, 0.792781537499611, -2.30137034971144, 1.16999587784412, 0.889469621266034,
            0.239667010403193, -0.969309214660315, 5.21461903375948, 0.348799845137009, -3.7347620794893,
            0.25771517766665 },
          { -2.13629241669652, -0.353986293257471, 9.61086066828255, -2.33661235049463, 1.93528728163342,
            -0.843722079096251, 0.999688424438972, -8.55711877917878, 1.02394998490757, -15.267288590286,
            -13.235596389738 },
          { 23.4650215184851, -0.726082462224456, 0.941255141296868, -1.25219178023961, 1.87830610438659,
            -0.0626703177011942, 0.798264835732211, 2.24872165119832, 0.499173544293671, -1.38568068487979,
            -2.73161981770289 },
          { 2.57560609203081, 0.129379326313481, -4.39493675912134, 0.755378151374909, -4.00108724778843,
            0.203477875797884, -1.08477900245397, -7.03054484748858, -1.09623688169448, -26.3844077677349,
            -41.2491346895123 },
          { -0.768854476049515, 0.0931802742811194, 0.0821972098693852, -3.89193022157862, 14.9873083975585,
            1.28580626432386, -0.181845844796471, -8.47241051331203, -0.943962517139566, 6.01103622570362,
            9.48437982003848 },
          { 0.178040151771032, 0.136134754761245, -0.787571684703651, 2.32182273660402, 7.30437794491034,
            -16.4278614062814, -0.113422629084682, -1.0680200019718, -0.568606303594401, 1.0249516698155,
            -7.95920000382392 },
          { 0.675546296267863, -0.385104312033727, -0.205638813395668, -3.31653776098415, 10.5616483640347,
            0.218068680685705, 0.605728183391577, -4.63802179325434, -0.254055135097754, 5.28618861484003,
            6.64663231117868 },
          { -4.21416802017974, -1.23106312717449, -1.75652044911551, -1.53276007675347, -1.36027716886005,
            -1.6755055391045, 0.843394761591634, 6.71332112562865, -3.21984188557587, -2.49871614350354,
            -5.16629424142249 },
          { -3.17947657486281, -2.70445335081074, 1.17896471286998, -1.88889268162012, 0.115966139146512,
            -1.69860246319196, -0.329404631248515, -6.09206258757561, 3.39801515225553, -3.44730923665625,
            -7.36048093738012 },
          { -12.1422669814385, -0.937172442798055, 2.04047673532808, -0.2149743778283, -2.6487701795534,
            -2.62694503937013, -0.530083657842696, -14.923114328578, -0.317336092090262, 23.1177068701846,
            6.21557901179705 },
          { 2.9112902347786, -4.39915513290615, -6.29077091232232, -3.50206937799633, -0.367072002518522,
            0.145186305220916, -6.21085587377794, 11.6250310635655, -7.43312502309394, 3.26005495282282,
            -11.6647818884921 } } };
    constexpr auto fWeightMatrix1to2 = std::array<float, 13>{
        { 0.737121712699059, -1.04254794966614, -0.789570860582197, 1.11549851568754, 1.5426749194788, -1.2610505044793,
          -0.630483186934049, 1.57472828085649, 0.790891919516845, 0.937883258898894, -0.781778561713509,
          -0.610089040707349, -0.385566736632017 } };

    // Normalization transformation
    constexpr auto transformer = TMV::Utils::Transformer{ fMin, fMax };

    // the training input variables
    constexpr auto validator = TMV::Utils::Validator{
        "ReadGhostNNVeloUT",
        std::tuple{ "log(abs(1./qop-1./qopUT))", "redChi2",
                    "abs((x+(zMagMatch-770.0)*tx)-(xEndT+(zMagMatch-9410.0)*txEndT))", "abs(ySeedMatch-yEndT)",
                    "abs(yParam0Final-yParam0Init)", "abs(yParam1Final-yParam1Init)", "abs(ty)", "abs(qop)", "abs(tx)",
                    "abs(xParam1Final-xParam1Init)" } };

    constexpr auto l0To1 = TMV::Utils::Layer{ fWeightMatrix0to1, ActivationFnc };
    constexpr auto l1To2 = TMV::Utils::Layer{ fWeightMatrix1to2, OutputActivationFnc };
    constexpr auto MVA   = TMV::Utils::MVA{ validator, transformer, 0, l0To1, l1To2 };
  } // namespace NNVeloUT

  struct ReadGhostNNVeloUT final {

    // constructor
    ReadGhostNNVeloUT( LHCb::span<const std::string_view, 10> theInputVars ) { NNVeloUT::MVA.validate( theInputVars ); }

    // the classifier response
    // "inputValues" is a vector of input values in the same order as the
    // variables given to the constructor
    static constexpr auto GetMvaValue( LHCb::span<const float, 10> input ) { return NNVeloUT::MVA( input ); }
  };

} // namespace LHCb::Pr::Forward

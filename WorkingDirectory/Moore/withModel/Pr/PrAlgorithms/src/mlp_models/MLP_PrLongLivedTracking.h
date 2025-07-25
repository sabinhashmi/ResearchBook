/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "../PrDownTrack.h"
#include "Event/PrSeedTracks.h"
#include "LHCbMath/FastMaths.h"
#include "LHCbMath/VectorizedML/Evaluators.h"
#include "LHCbMath/VectorizedML/Feature.h"
#include "LHCbMath/VectorizedML/Sequence.h"

namespace MLP::PrLongLivedTracking {
  // We need a struct to contains both UT segment and SciFi segment information as input, and also to store the output
  struct DataType_t {
    // Input
    PrDownTrack*                     m_downstream_track;
    const LHCb::Pr::Seeding::Tracks* m_seed_container;
    const int                        m_seed_idx;
    // Output
    float ghost_probability = std::numeric_limits<float>::quiet_NaN();

    // Constructor
    template <typename SeedProxy_t>
    DataType_t( PrDownTrack& downstream_track, SeedProxy_t scifi_seed )
        : m_downstream_track( &downstream_track )
        , m_seed_container( scifi_seed.container() )
        , m_seed_idx( scifi_seed.offset() ) {}

    // Useful functions
    auto ft_proxy() const { return *( m_seed_container->scalar().begin() + m_seed_idx ); }

    // Interfaces
    float ut_z() const { return m_downstream_track->zUT(); }

    float ut_x() const { return m_downstream_track->xAtZ( ut_z() ); }

    float ut_y() const { return m_downstream_track->yAtZ( ut_z() ); }

    float ut_tx() const { return m_downstream_track->slopeX(); }

    float ut_ty() const { return m_downstream_track->slopeY(); }

    float ut_chi2() const { return m_downstream_track->chi2(); }

    float ut_nhits() const { return m_downstream_track->hits().size(); }

    float ft_z() const { return m_downstream_track->stateZ(); }

    float ft_x() const { return m_downstream_track->stateX(); }

    float ft_y() const { return m_downstream_track->stateY(); }

    float ft_tx() const { return m_downstream_track->stateTx(); }

    float ft_ty() const { return m_downstream_track->stateTy(); }

    float ft_chi2() const { return ft_proxy().chi2PerDoF().cast(); }

    float ft_nhits() const { return ft_proxy().nFTHits().cast(); }

    float eta() const { return std::asinh( 1.f / std::hypot( ut_tx(), ut_ty() ) ); }

    float dtx() const { return ft_tx() - ut_tx(); }

    float dy() const { return ft_y() - ( ut_y() + ut_ty() * ( ft_z() - ut_z() ) ); }
  };

} // namespace MLP::PrLongLivedTracking

namespace MLP::PrLongLivedTracking::FeatureDefinition {
#define PRLONGLIVEDTRACKING_DECLARE_FEATURE( FEATURE_NAME )                                                            \
  struct FEATURE_NAME : LHCb::VectorizedML::Feature {                                                                  \
    const char* name() const { return m_name; }                                                                        \
    float       operator()( const DataType_t& input_track ) const { return input_track.FEATURE_NAME(); }               \
                                                                                                                       \
  private:                                                                                                             \
    static constexpr auto m_name = #FEATURE_NAME;                                                                      \
  };

  // Simplify the declaration
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_x )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_y )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_tx )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_ty )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_chi2 )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ut_nhits )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_x )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_y )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_tx )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_ty )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_chi2 )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( ft_nhits )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( eta )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( dtx )
  PRLONGLIVEDTRACKING_DECLARE_FEATURE( dy )
} // namespace MLP::PrLongLivedTracking::FeatureDefinition

namespace MLP::PrLongLivedTracking {
  using Features =
      LHCb::VectorizedML::Features<FeatureDefinition::ut_x, FeatureDefinition::ut_y, FeatureDefinition::ut_tx,
                                   FeatureDefinition::ut_ty, FeatureDefinition::ut_chi2, FeatureDefinition::ut_nhits,
                                   FeatureDefinition::ft_chi2, FeatureDefinition::ft_nhits, FeatureDefinition::eta,
                                   FeatureDefinition::dtx, FeatureDefinition::dy>;

  // Define network
  constexpr auto nInput   = Features::Size;
  constexpr auto nNeurons = 32u;
  constexpr auto nOutput  = 1u;
  struct P0 : LHCb::VectorizedML::Layers::StandardScaler<nInput> {};
  struct L1 : LHCb::VectorizedML::Layers::Linear<nInput, nNeurons> {};
  struct A2 : LHCb::VectorizedML::Layers::ReLU<nNeurons> {};
  struct L3 : LHCb::VectorizedML::Layers::Linear<nNeurons, nOutput> {};
  struct A4 : LHCb::VectorizedML::Layers::Sigmoid<nOutput> {};

  struct Sequence : LHCb::VectorizedML::Sequence<nInput, nOutput, P0, L1, A2, L3, A4> {};
  struct Evaluator : LHCb::VectorizedML::EvaluatorOnRange<Sequence, Features> {};
} // namespace MLP::PrLongLivedTracking

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

#include "Event/UTHitCluster.h"
#include <PrKernel/UTHitHandler.h>

#include <UTDAQ/UTDAQHelper.h>
#include <UTDet/DeUTDetector.h>
#include <UTDet/DeUTSector.h>

#include "LHCbAlgs/Transformer.h"

#include "Detector/UT/UTConstants.h"

namespace LHCb {
  /**
   *  Transform the "UT::Hit" objects to a set of UTHitClusters.
   *  Effectively removing any geometry-dependent information from
   *  the hits, leaving only the bare minimum to reproduce these
   *  hits given a geometry, and going to a finite numerical
   *  precision.
   *
   *  @author Laurent Dufour
   */
  struct UTHitHandlerToUTHitClusterConverter : Algorithm::Transformer<UTHitClusters( ::UT::HitHandler const& )> {

    UTHitHandlerToUTHitClusterConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputHits", "" } }, { KeyValue{ "OutputClusters", "" } } ) {}
    UTHitClusters operator()( ::UT::HitHandler const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::UTHitHandlerToUTHitClusterConverter, "UTHitHandlerToUTHitClusterConverter" )

LHCb::UTHitClusters LHCb::UTHitHandlerToUTHitClusterConverter::operator()( ::UT::HitHandler const& dst ) const {
  LHCb::UTHitClusters outputClusters;
  outputClusters.reserve( dst.hits().size() );

  // While the HitHandler is ordered, it's not sorted
  // this means that the hits are stored in blocks of geomIdx,
  // but these blocks are not sorted among each other.
  // meanwhile, this is expected by the IndexedHitContainer.
  // therefore, the clusters are inserted block-by-block
  // into the IHC, starting with the lowest index

  const auto& all_ut_hits = dst.hits();

  for ( unsigned int uniqueSector = 0; uniqueSector < LHCb::Detector::UT::MaxNumberOfSectors; ++uniqueSector ) {
    const auto& indices = dst.indices( uniqueSector );
    if ( indices.first == indices.second ) continue;

    std::for_each( all_ut_hits.begin() + indices.first, all_ut_hits.begin() + indices.second,
                   [&outputClusters]( const auto& ut_hit ) {
                     outputClusters.addHit( ut_hit.chanID(), ut_hit.fracStrip(), ut_hit.size(),
                                            ut_hit.clusterCharge() );
                   } );
  }

  outputClusters.setOffsets();

  m_convertedClusters += outputClusters.size();

  return outputClusters;
}

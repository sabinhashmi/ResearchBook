/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/Track.h"
#include "LHCbAlgs/ScalarTransformer.h"

/** @class fromVectorLHCbTrack fromVectorLHCbTrack.h
 *
 *  Small helper to convert std::vector<LHCb::Track> to LHCb::Tracks
 *
 *  @author Renato Quagliani, Michel De Cian
 *  @date   2018-03-09
 */

namespace LHCb::Converters::Track::v1 {

  struct fromVectorLHCbTrack
      : public Algorithm::ScalarTransformer<fromVectorLHCbTrack, Tracks( const std::vector<LHCb::Track>& )> {

    fromVectorLHCbTrack( const std::string& name, ISvcLocator* pSvcLocator )
        : ScalarTransformer( name, pSvcLocator, KeyValue{ "InputTracksName", "" },
                             KeyValue{ "OutputTracksName", "" } ) {}

    using ScalarTransformer::operator();

    /// The main function, converts the track
    LHCb::Track* operator()( const LHCb::Track& track ) const {
      return track.hasKey() ? new LHCb::Track( track, track.key() ) : new LHCb::Track( track );
    }
  };
  DECLARE_COMPONENT( fromVectorLHCbTrack )
} // namespace LHCb::Converters::Track::v1

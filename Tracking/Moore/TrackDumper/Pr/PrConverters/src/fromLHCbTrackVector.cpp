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
#ifndef LHCB_CONVERTERS_TRACK_V1_FROMLHCBTRACKVECTOR_H
#define LHCB_CONVERTERS_TRACK_V1_FROMLHCBTRACKVECTOR_H

// Include files
// from Gaudi
#include "Event/Track.h"
#include "LHCbAlgs/Transformer.h"

/** @class fromLHCbTrackVector fromLHCbTrackVector.h
 *
 *  Small helper to convert LHCb::Tracks to std::vector<LHCb::Track>
 *
 */

namespace LHCb {
  namespace Converters {
    namespace Track {
      namespace v1 {
        class fromLHCbTrackVector : public Algorithm::Transformer<std::vector<LHCb::Track>( const LHCb::Tracks& )> {
        public:
          fromLHCbTrackVector( const std::string& name, ISvcLocator* pSvcLocator )
              : Transformer( name, pSvcLocator, KeyValue{ "InputTracksName", "" },
                             KeyValue{ "OutputTracksName", "" } ) {}
          // The main function, converts the track
          std::vector<LHCb::Track> operator()( const LHCb::Tracks& inputTracks ) const override {
            std::vector<LHCb::Track> outputTracks;
            outputTracks.reserve( inputTracks.size() );
            for ( const auto& track : inputTracks ) {
              if ( track->hasKey() ) {
                outputTracks.emplace_back( *track, track->key() );
              } else {
                outputTracks.emplace_back( *track );
              }
            }
            return outputTracks;
          }
        };
        DECLARE_COMPONENT( fromLHCbTrackVector )
      } // namespace v1
    }   // namespace Track
  }     // namespace Converters
} // namespace LHCb
#endif // LHCB_CONVERTERS_TRACK_V1_FROMLHCBTRACKVECTOR_H

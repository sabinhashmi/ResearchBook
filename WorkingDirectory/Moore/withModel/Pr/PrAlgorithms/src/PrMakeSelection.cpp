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
#include "Gaudi/Accumulators.h"
#include "LHCbAlgs/Transformer.h"
#include "PrKernel/PrSelection.h"
#include <vector>

namespace Pr {
  // Helper algorithm that makes a Selection<T> from a Container
  /** @class MakeSelection PrMakeSelection.cpp
   *
   *  MakeSelection<T, Container> creates a Selection<T> object pointing at the contiguous storage object Container
   *  and flags all elements as selected.
   *
   *  @tparam T         The selected object type (e.g. Track, Particle, ...). By contruction this is not copied, as the
   *                    output type Selection<T> is just a view of the underlying storage Container.
   *  @tparam Container Sets the type of the underlying storage of T. This must be convertible to LHCb::span<T const>
   *                    and is std::vector<T> by default.
   */
  template <typename T, typename Container = std::vector<T> const&>
  struct MakeSelection final : public LHCb::Algorithm::Transformer<Selection<T>( Container )> {
    MakeSelection( const std::string& name, ISvcLocator* pSvcLocator )
        : LHCb::Algorithm::Transformer<Selection<T>( Container )>( name, pSvcLocator, { "Input", "" },
                                                                   { "Output", "" } ) {}

    Selection<T> operator()( Container in ) const override {
      m_inputObjects += in.size();
      return { in }; // by default we get a selection with everything marked accepted
    }

    mutable Gaudi::Accumulators::Counter<> m_inputObjects{ this, "# input objects" };
  };

  DECLARE_COMPONENT_WITH_ID( MakeSelection<LHCb::Event::v1::Track>, "MakeSelection__Track_v1" )
} // namespace Pr

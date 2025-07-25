/*****************************************************************************\
* (c) Copyright 2000-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

#include <array>
#include <limits>
#include <type_traits>

namespace Hough {
  // pow that can be used in constexpr contexts
  inline constexpr int constpow( const int base, const unsigned int exponent ) {
    // (parentheses not required in next line)
    return ( exponent == 0 ) ? 1 : ( base * constpow( base, exponent - 1 ) );
  }

  // Static looper
  template <int N>
  struct Looper {
    template <typename NC, typename F, typename... Ts>
    constexpr void operator()( NC& n, F& f, Ts... x ) {
      size_t i = 0;
      do {
        if constexpr ( N == 0 ) {
          f( i, x... );
        } else {
          Looper<N - 1>()( n, f, i, x... );
        }
        ++i;
      } while ( n( N, i ) );
    }
  };

  template <typename ITERTYPE>
  struct always_true {
    inline constexpr bool operator()( const ITERTYPE& ) { return true; }
  };

  template <typename... T>
  struct always_false : std::false_type {};

  /** @class HoughSearch HoughSearch.h
   *  Class for Hough cluster search
   *
   *  - T         : Type of the underlying objects (e.g. a class representing the track detector hits)
   *  - DEPTH     : Maximum depth of the cluster search per layer; if N hits from the same layer can form a cluster at
   * the same position, the search will stop after DEPTH hits.
   *  - MAXNCAND  : If clusters with the largest number of hits are found at multiple positions, the search will stop
   * after MAXNCAND. This implies that the maximum possible number of candidates in the final output is MAXNRESULTS =
   * MAXNCAND*(DEPTH^NLAYERS).
   *  - NLAYERS   : The number of layers (== the maximum number of elements in a single Hough cluster).
   *  - NBINS...  : Parameter pack with the number of bins in each hypothesis. Each hypothesis correspond to a different
   * way of "interpreting" the hits. For example, there can be different hypothesis for the track origin, which imply a
   * different mapping from hit y position to y-z slope.
   *
   *  The algorithm starts by distributing the hits in NLAYERS histograms for each hypothesis. Histograms are
   * represented by fixed-size arrays. The NLAYERS histograms from the same hypothesis are added together to found the
   * Hough clusters. The Hough clusters with the largest number of elements (maximum is == NLAYERS) are promoted to
   * candidates and returned by the algorithm.
   *
   *  @author Salvatore Aiola (salvatore.aiola@cern.ch)
   *  @date   2020-02-26
   */
  template <typename T, int DEPTH, int MAXNCAND, int NLAYERS, int... NBINS>
  class HoughSearch {
  private:
    static_assert( ( ( NBINS > 0 ) && ... ), "The number of bins must be > 0 for every hypothesis." );

  public:
    static constexpr int MAXNRESULTS = MAXNCAND * constpow( DEPTH, NLAYERS );
    using result_type                = std::array<std::array<T, NLAYERS>, MAXNRESULTS>;

    template <typename Type>
    static constexpr Type invalid_value() {
      if constexpr ( std::is_integral_v<Type> ) {
        if constexpr ( std::is_unsigned_v<Type> ) return std::numeric_limits<Type>::max();
        return -1;
      } else if constexpr ( std::is_floating_point_v<Type> ) {
        return std::numeric_limits<Type>::signaling_NaN();
      } else if constexpr ( std::is_pointer_v<Type> ) {
        return nullptr;
      } else {
        static_assert( always_false<Type>::value &&
                       "No known invalid value for element type, please provide or use pointer to element type." );
      }
    }

    template <typename Type>
    static constexpr auto invalid( Type elem ) {
      return elem == invalid_value<Type>();
    }

    /** @brief Constructor without hypothesis-dependent factor array
     *  @param th       Minimum size of a Hough cluster to be considered as a candidate
     *  @param xmin     Low edge of the first bin of the histograms for each hypothesis
     *  @param binw     Bin width of the histograms for each hypothesis
     */
    HoughSearch( int th, std::array<float, sizeof...( NBINS )> xmin, std::array<float, sizeof...( NBINS )> binw )
        : m_th{ th }, m_binw{ binw }, m_n_valid_hits{ { { 0 } } }, m_clusters{ { 0 } } {
      for ( size_t h = 0; h < sizeof...( NBINS ); ++h ) {
        // Normalize by the bin width to simplify algebra later
        m_factors[h].fill( 1.f / binw[h] );
        m_xmin[h] = xmin[h] / binw[h];
      }
    }

    /** @brief Constructor without hypothesis-dependent factor array
     *  @param th       Minimum size of a Hough cluster to be considered as a candidate
     *  @param xmin     Low edge of the first bin of the histograms (the same for all hypothesis)
     *  @param binw     Bin width of the histograms (the same for all hypothesis)
     */
    HoughSearch( int th, float xmin, float binw )
        : m_th{ th }, m_binw{ binw }, m_n_valid_hits{ { { 0 } } }, m_clusters{ { 0 } } {
      float inv_binw = 1.f / binw;
      xmin *= inv_binw;
      for ( size_t h = 0; h < sizeof...( NBINS ); ++h ) {
        m_binw[h] = binw;
        // Normalize by the bin width to simplify algebra later
        m_factors[h].fill( inv_binw );
        m_xmin[h] = xmin;
      }
    }

    /** @brief Constructor with hypothesis-dependent factor array
     *  @param th       Minimum size of a Hough cluster to be considered as a candidate
     *  @param xmin     Low edge of the first bin of the histograms for each hypothesis
     *  @param binw     Bin width of the histograms for each hypothesis
     *  @param factors  The values of each elements will be multiplied by these factors to find the histogram bin.
     *                  This parameter is used to differentiate between hypothesis.
     */
    HoughSearch( int th, std::array<float, sizeof...( NBINS )> xmin, std::array<float, sizeof...( NBINS )> binw,
                 std::array<std::array<float, NLAYERS>, sizeof...( NBINS )> factors )
        : m_th{ th }, m_binw{ binw }, m_n_valid_hits{ { { 0 } } }, m_clusters{ { 0 } } {
      for ( size_t h = 0; h < sizeof...( NBINS ); ++h ) {
        // Normalize by the bin width to simplify algebra later
        std::transform( factors[h].begin(), factors[h].end(), m_factors[h].begin(),
                        [&binw, &h]( float f ) -> float { return f / binw[h]; } );
        m_xmin[h] = xmin[h] / binw[h];
      }
    }

    /** @brief Constructor with hypothesis-dependent factor array
     *  @param th       Minimum size of a Hough cluster to be considered as a candidate
     *  @param xmin     Low edge of the first bin of the histograms (the same for all hypothesis)
     *  @param binw     Bin width of the histograms (the same for all hypothesis)
     *  @param factors  The values of each elements will be multiplied by these factors to find the histogram bin.
     *                  This parameter is used to differentiate between hypothesis.
     */
    HoughSearch( int th, float xmin, float binw, std::array<std::array<float, NLAYERS>, sizeof...( NBINS )> factors )
        : m_th( th ), m_binw{}, m_n_valid_hits{ { { 0 } } }, m_clusters{ { 0 } } {
      float inv_binw = 1.f / binw;
      xmin *= inv_binw;
      for ( size_t h = 0; h < sizeof...( NBINS ); ++h ) {
        m_binw[h] = binw;
        // Normalize by the bin width to simplify algebra later
        std::transform( factors[h].begin(), factors[h].end(), m_factors[h].begin(),
                        [&inv_binw]( float f ) -> float { return f * inv_binw; } );
        m_xmin[h] = xmin;
      }
    }

    /** @brief Recursive-template method to add an element to the histograms for each hypothesis
     *  @param layer            The layer index to which the element belongs
     *  @param x                The value of the element (e.g. teh hit position)
     *  @param elem             Pointer to the element (e.g. a class representing a detector hit)
     *  @param nbins            The number of bins in the first hypothesis
     *  @param nbins_remaining  The number of bins in the remaining hypothesis with which this method will be called
     * recursively until empty.
     */
    template <typename... Ts>
    void add_by_hypothesis( int layer, float x, T elem, int nbins, Ts... nbins_remaining ) {
      // CAUTION: layer index is not checked!
      // Hypothesis index is the different between the total number of hypothesis and the number of remaining hypothesis
      constexpr size_t h = sizeof...( NBINS ) - sizeof...( nbins_remaining ) - 1;
      // Calculate and check for boundaries the bin index
      int bin = x * m_factors[h][layer] - m_xmin[h];
      if ( bin < 0 || bin >= nbins ) return;
      // Apply the bin shift of the current hypothesis
      bin += NCUMBINS<h>;
      // Skip this element if the maximum depth was reached for this bin
      if ( m_n_valid_hits[layer][bin] >= DEPTH ) return;
      // Store the pointer to the element
      m_hits[bin][layer][m_n_valid_hits[layer][bin]] = elem;
      // Increase the depth of this bin
      ++m_n_valid_hits[layer][bin];
      // Increase the cluster size at this bin position
      if ( m_n_valid_hits[layer][bin] == 1 ) ++m_clusters[bin];
      // Call again this method for the remaining hypothesis
      if constexpr ( sizeof...( nbins_remaining ) > 0 ) add_by_hypothesis( layer, x, elem, nbins_remaining... );
    }

    /** @brief Add an element to the histograms for all hypothesis
     *  @param layer            The layer index to which the element belongs
     *  @param x                The value of the element (e.g. teh hit position)
     *  @param elem             Pointer to the element (e.g. a class representing a detector hit)
     */
    void add( int layer, float x, T elem ) { add_by_hypothesis( layer, x, elem, NBINS... ); }

    /** @brief Serialize the candidates which are implicitly defined by the depths of the bins in each layer
     *  @param result_iter      Iterator of a collection where the candidates (arrays of T pointers) are returned
     *  @param min_total_layers Optional integer specifying how many total layers must be present after checking
     *                          the neighbours for missing layers in the raw search.
     *  @param check            Optional functor to be applied to the iterator. If the functor returns false,
     *                          the search is stopped. For example, if using an array smaller than the
     *                          maximum possible number of candidates (MAXNRESULTS) then this function
     *                          should check that the end of the array has been reached.
     */
    template <typename ITERTYPE, typename CHECKITER = always_true<ITERTYPE>>
    ITERTYPE search( ITERTYPE result_iter, int min_total_layers = 0, CHECKITER check = always_true<ITERTYPE>() ) const {
      raw_candidates_type       raw_candidates;
      depth_raw_candidates_type raw_candidate_depth;
      int                       n_raw_candidates = search_raw( raw_candidates, raw_candidate_depth, min_total_layers );
      // Loop over the "raw" candidates.
      // Raw candidates may include multiple implicit candidates,
      // if the depths of one or more layer is > 1
      for ( int iraw = 0; iraw < n_raw_candidates; ++iraw ) {
        // This lambda function is the body of the depth static loop
        // i are integers corresponding to the depths in each layer
        // if no element is present for a layer the depth for that layer is always zero (and the corresponding element
        // is nullptr) if a layer has multiple elements (total depth > 1), than multiple iterations are done, each time
        // with a different depth for that layer
        auto f = [&]( auto... i ) {
          std::array<size_t, NLAYERS> depths{ i... };
          int                         ilayer = 0, ihit = 0;
          for ( const auto& d : depths ) {
            if ( d < raw_candidate_depth[iraw][ilayer] ) {
              // A hit is present in this layer
              ( *result_iter )[ihit] = raw_candidates[iraw][ilayer][d];
              ++ihit;
            } else {
              // No hit in this layer. Zero-ing one position at the end of the array to adapt the total length of the
              // cluster. This is needed, because the array is not guaranteed to be initialized and not doing so may
              // result in the last element(s) of the array remaining uninitialized.
              ( *result_iter )[NLAYERS - 1 - ilayer + ihit] = invalid_value<T>();
            }
            ++ilayer;
          }
          ++result_iter;
        };
        // This lambda function is used to stop the search after the total depth of a layer has been reached
        // or the result iterator is not valid anymore
        auto nc = [&]( int ilayer, size_t d ) {
          return ( d < raw_candidate_depth[iraw][ilayer] && check( result_iter ) );
        };
        // This initiates a static loop through the depths of each layer
        Looper<NLAYERS - 1>()( nc, f );
      }
      return result_iter;
    }

  private:
    using raw_candidates_type       = std::array<std::array<std::array<T, DEPTH>, NLAYERS>, MAXNCAND>;
    using depth_raw_candidates_type = std::array<std::array<size_t, NLAYERS>, MAXNCAND>;

    /** @brief Search the largest Hough cluster(s)
     *  @param result           Reference to the array where the result of the search are returned
     *  @param depth            Reference to the array where the total depths in each layer are returned
     *  @param min_total_layers Minimum of total layers (after checking neighbour bins) to accecpt candidate
     */
    int search_raw( raw_candidates_type& result, depth_raw_candidates_type& depth, int min_total_layers = 0 ) const {
      // Compute the index of the first cluster with largest size
      auto max_elem = std::max_element( m_clusters.begin(), m_clusters.end() );
      // If the size of the largest cluster is below threshold return
      if ( *max_elem < m_th ) return 0;
      int ibin   = max_elem - m_clusters.begin();
      int n_cand = 0;
      // Loop through all bins (== cluster), starting from the first large cluster
      do {
        int tot_layers = 0;
        // Only consider bins (clusters) that are as large as the largest
        if ( m_clusters[ibin] == *max_elem ) {
          result[n_cand] = m_hits[ibin];
          for ( int ilayer = 0; ilayer < NLAYERS; ++ilayer ) {
            depth[n_cand][ilayer] = m_n_valid_hits[ilayer][ibin];
            if ( depth[n_cand][ilayer] == 0 ) {
              // If no element is found in the current layer, try to look for possible
              // candidates in the previous or next bin that may extend the cluster.
              int           d1         = 0;
              constexpr int half_depth = DEPTH / 2 + DEPTH % 2;
              if ( !is_lower_bound_bin( ibin ) ) {
                for ( size_t d2 = 0; d2 < m_n_valid_hits[ilayer][ibin - 1]; ++d2 ) {
                  result[n_cand][ilayer][d1] = m_hits[ibin - 1][ilayer][d2];
                  ++d1;
                  if ( d1 >= half_depth ) break;
                }
              }
              if ( !is_upper_bound_bin( ibin ) ) {
                for ( size_t d2 = 0; d2 < m_n_valid_hits[ilayer][ibin + 1]; ++d2 ) {
                  result[n_cand][ilayer][d1] = m_hits[ibin + 1][ilayer][d2];
                  ++d1;
                  if ( d1 >= DEPTH ) break;
                }
              }
              depth[n_cand][ilayer] = d1;
              tot_layers += ( d1 > 0 );
            } else {
              ++tot_layers;
            }
          }
          n_cand += ( tot_layers >= min_total_layers );
        }
        ++ibin;
      } while ( ibin < NTOTBINS && n_cand < MAXNCAND );
      return n_cand;
    }

  private:
    static inline constexpr std::array<int, sizeof...( NBINS )> NCUMBINS_ARRAY( int seed = 0 ) {
      return { { seed += NBINS... } };
    }

    template <int N>
    static constexpr int NCUMBINS = N == 0 ? 0 : NCUMBINS_ARRAY()[N - 1];

    static constexpr int NTOTBINS = NCUMBINS_ARRAY()[sizeof...( NBINS ) - 1];

    template <size_t... N>
    static inline constexpr bool is_upper_bound_bin_impl( const int bin, std::index_sequence<N...> ) {
      return ( ( bin == ( NCUMBINS<N + 1> - 1 ) ) || ... );
    }

    template <size_t... N>
    static inline constexpr bool is_lower_bound_bin_impl( const int bin, std::index_sequence<N...> ) {
      return ( ( bin == (NCUMBINS<N>)) || ... );
    }

    template <typename Indices = std::make_index_sequence<sizeof...( NBINS )>>
    static inline constexpr bool is_upper_bound_bin( const int bin ) {
      return is_upper_bound_bin_impl( bin, Indices{} );
    }

    template <typename Indices = std::make_index_sequence<sizeof...( NBINS )>>
    static inline constexpr bool is_lower_bound_bin( const int bin ) {
      return is_lower_bound_bin_impl( bin, Indices{} );
    }

  private:
    int                                   m_th;   // Minimum size of Hough cluster
    std::array<float, sizeof...( NBINS )> m_xmin; // Lower edge of the first bin of the histograms for each hypothesis
    std::array<float, sizeof...( NBINS )> m_binw; // Bin width of the histograms for each hypothesis
    std::array<std::array<float, NLAYERS>, sizeof...( NBINS )> m_factors;   // Bin width of the histograms for each
                                                                            // hypothesis
    std::array<std::array<std::array<T, DEPTH>, NLAYERS>, NTOTBINS> m_hits; // Pointers to the elements (class T
                                                                            // represents the underlying objects e.g.
                                                                            // detector hits)
    std::array<std::array<size_t, NTOTBINS>, NLAYERS> m_n_valid_hits; // Number of valid hits (total actual depth) for
                                                                      // each bin in each layer
    std::array<int, NTOTBINS> m_clusters;                             // Size of the clusters corresponding to each bin
  };
} // namespace Hough

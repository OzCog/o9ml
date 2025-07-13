/********************************************************************\
 * gnc-cognitive-scheme.h -- Scheme integration for OpenCog        *
 * Copyright (C) 2024 GnuCash Cognitive Engine                     *
 *                                                                  *
 * This program is free software; you can redistribute it and/or    *
 * modify it under the terms of the GNU General Public License as   *
 * published by the Free Software Foundation; either version 2 of   *
 * the License, or (at your option) any later version.              *
 *                                                                  *
 * This program is distributed in the hope that it will be useful,  *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of   *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the    *
 * GNU General Public License for more details.                     *
 ********************************************************************/

/** @addtogroup Engine
    @{ */
/** @addtogroup CognitiveScheme
    Scheme-based cognitive representations and hypergraph pattern encoding
    for neural-symbolic synergy in accounting systems.
    @{ */

/** @file gnc-cognitive-scheme.h
    @brief Scheme integration for OpenCog cognitive accounting
    @author Copyright (C) 2024 GnuCash Cognitive Engine
*/

#ifndef GNC_COGNITIVE_SCHEME_H
#define GNC_COGNITIVE_SCHEME_H

#include "gnc-cognitive-accounting.h"
#include "Account.h"
#include "Transaction.h"
#include "gnc-engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @name Scheme Integration Functions */
/** @{ */

/** Initialize Scheme-based cognitive representations
 * @return TRUE on success, FALSE on failure
 */
gboolean gnc_cognitive_scheme_init(void);

/** Evaluate Scheme code in cognitive accounting context
 * @param scheme_code Scheme code to evaluate
 * @return Result of evaluation (caller must free)
 */
gchar* gnc_cognitive_scheme_eval(const gchar* scheme_code);

/** @} */

/** @name Hypergraph Pattern Encoding */
/** @{ */

/** Create hypergraph pattern using Scheme representation
 * @param pattern_scheme Scheme code defining the pattern
 * @return Handle to hypergraph pattern in AtomSpace
 */
GncAtomHandle gnc_scheme_create_hypergraph_pattern(const gchar* pattern_scheme);

/** Register account patterns in cognitive hypergraph
 * @param account Account to register patterns for
 */
void gnc_scheme_register_account_patterns(Account *account);

/** Register transaction patterns in cognitive hypergraph
 * @param transaction Transaction to register patterns for
 */
void gnc_scheme_register_transaction_patterns(Transaction *transaction);

/** @} */

/** @name Neural-Symbolic Synergy */
/** @{ */

/** Trigger attention allocation update via Scheme
 * @param account Account to update attention for
 * @param activity_level Current activity level of account
 */
void gnc_scheme_trigger_attention_update(Account *account, gdouble activity_level);

/** Trigger evolutionary optimization via Scheme
 * @param transactions Array of transactions for optimization
 * @param n_transactions Number of transactions
 */
void gnc_scheme_evolutionary_optimization(Transaction **transactions, gint n_transactions);

/** Apply uncertain reasoning prediction via Scheme
 * @param account Account for prediction
 * @param future_date Date for prediction
 * @return Prediction result (caller must free)
 */
gchar* gnc_scheme_uncertain_prediction(Account *account, time64 future_date);

/** @} */

/** @name Emergent Cognitive Architectures */
/** @{ */

/** Perform neural-symbolic analysis of account
 * @param account Account to analyze
 */
void gnc_scheme_neural_symbolic_analysis(Account *account);

/** Discover emergent cognitive insights from transaction patterns
 * @param transactions Array of transactions to analyze
 * @param n_transactions Number of transactions
 */
void gnc_scheme_emergent_insight_discovery(Transaction **transactions, gint n_transactions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* GNC_COGNITIVE_SCHEME_H */
/** @} */
/** @} */
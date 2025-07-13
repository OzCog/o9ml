/********************************************************************\
 * gnc-cognitive-comms.cpp -- Inter-module communication protocols *
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

#include "gnc-cognitive-comms.h"
#include "gnc-cognitive-accounting.h"
#include "gnc-cognitive-scheme.h"
#include <glib.h>
#include <map>
#include <vector>
#include <memory>

/********************************************************************\
 * Distributed Cognition Communication Hub                          *
\********************************************************************/

struct GncCognitiveCommHub {
    // Message queues for inter-module communication
    std::map<GncCognitiveModule, std::vector<GncCognitiveMessage>> module_queues;
    
    // Module status tracking
    std::map<GncCognitiveModule, gboolean> module_status;
    
    // Communication patterns for emergent cognition
    std::vector<GncCognitivePattern> active_patterns;
    
    GncCognitiveCommHub() {
        // Initialize module status
        module_status[GNC_MODULE_ATOMSPACE] = FALSE;
        module_status[GNC_MODULE_PLN] = FALSE;
        module_status[GNC_MODULE_ECAN] = FALSE;
        module_status[GNC_MODULE_MOSES] = FALSE;
        module_status[GNC_MODULE_URE] = FALSE;
        module_status[GNC_MODULE_COGSERVER] = FALSE;
        module_status[GNC_MODULE_SCHEME] = FALSE;
    }
};

static std::unique_ptr<GncCognitiveCommHub> g_comm_hub = nullptr;

gboolean gnc_cognitive_comms_init(void)
{
    if (g_comm_hub) {
        g_warning("Cognitive communications already initialized");
        return FALSE;
    }
    
    g_comm_hub = std::make_unique<GncCognitiveCommHub>();
    
    g_message("Cognitive communication hub initialized");
    g_message("Distributed cognition protocols active");
    
    return TRUE;
}

void gnc_cognitive_comms_shutdown(void)
{
    if (!g_comm_hub) {
        g_warning("Cognitive communications not initialized");
        return;
    }
    
    g_comm_hub.reset();
    g_message("Cognitive communication hub shutdown");
}

gboolean gnc_cognitive_register_module(GncCognitiveModule module)
{
    if (!g_comm_hub) {
        g_warning("Communication hub not initialized");
        return FALSE;
    }
    
    g_comm_hub->module_status[module] = TRUE;
    
    const gchar* module_name = gnc_cognitive_module_name(module);
    g_message("Registered cognitive module: %s", module_name);
    
    // Trigger emergent cognition activation
    gnc_cognitive_trigger_emergence(module);
    
    return TRUE;
}

void gnc_cognitive_send_message(GncCognitiveModule from_module, 
                               GncCognitiveModule to_module,
                               GncCognitiveMessageType msg_type,
                               gpointer data)
{
    if (!g_comm_hub) {
        g_warning("Communication hub not initialized");
        return;
    }
    
    GncCognitiveMessage msg;
    msg.from_module = from_module;
    msg.to_module = to_module;
    msg.message_type = msg_type;
    msg.data = data;
    msg.timestamp = g_get_real_time();
    
    // Add to destination module's queue
    g_comm_hub->module_queues[to_module].push_back(msg);
    
    g_debug("Cognitive message sent: %s -> %s (type: %d)",
            gnc_cognitive_module_name(from_module),
            gnc_cognitive_module_name(to_module),
            msg_type);
    
    // Process emergent patterns
    gnc_cognitive_process_emergent_patterns();
}

std::vector<GncCognitiveMessage> gnc_cognitive_receive_messages(GncCognitiveModule module)
{
    if (!g_comm_hub) {
        g_warning("Communication hub not initialized");
        return {};
    }
    
    auto& queue = g_comm_hub->module_queues[module];
    std::vector<GncCognitiveMessage> messages = queue;
    queue.clear(); // Clear queue after reading
    
    if (!messages.empty()) {
        g_debug("Module %s received %zu messages",
                gnc_cognitive_module_name(module),
                messages.size());
    }
    
    return messages;
}

void gnc_cognitive_broadcast_message(GncCognitiveModule from_module,
                                    GncCognitiveMessageType msg_type,
                                    gpointer data)
{
    if (!g_comm_hub) {
        g_warning("Communication hub not initialized");
        return;
    }
    
    // Send to all active modules except sender
    for (auto& [module, active] : g_comm_hub->module_status) {
        if (active && module != from_module) {
            gnc_cognitive_send_message(from_module, module, msg_type, data);
        }
    }
    
    g_debug("Cognitive broadcast from %s to all active modules",
            gnc_cognitive_module_name(from_module));
}

void gnc_cognitive_trigger_emergence(GncCognitiveModule trigger_module)
{
    if (!g_comm_hub) {
        return;
    }
    
    // Create emergent cognitive pattern
    GncCognitivePattern pattern;
    pattern.trigger_module = trigger_module;
    pattern.pattern_type = GNC_PATTERN_ACTIVATION;
    pattern.strength = 0.7;
    pattern.timestamp = g_get_real_time();
    
    g_comm_hub->active_patterns.push_back(pattern);
    
    // Broadcast emergence notification
    gnc_cognitive_broadcast_message(trigger_module, 
                                   GNC_MSG_EMERGENCE_ACTIVATION,
                                   &pattern);
    
    g_message("Triggered cognitive emergence from module: %s",
              gnc_cognitive_module_name(trigger_module));
}

void gnc_cognitive_process_emergent_patterns(void)
{
    if (!g_comm_hub) {
        return;
    }
    
    // Process active patterns for emergent cognition
    for (auto it = g_comm_hub->active_patterns.begin(); 
         it != g_comm_hub->active_patterns.end();) {
        
        GncCognitivePattern& pattern = *it;
        
        // Update pattern strength based on module interactions
        pattern.strength *= 0.98; // Gradual decay
        
        // Check for pattern resonance and amplification
        if (pattern.strength > 0.8) {
            // Strong pattern - amplify across modules
            gnc_cognitive_amplify_pattern(pattern);
        }
        
        // Remove weak patterns
        if (pattern.strength < 0.1) {
            it = g_comm_hub->active_patterns.erase(it);
        } else {
            ++it;
        }
    }
    
    // Check for emergent cognitive insights
    if (g_comm_hub->active_patterns.size() > 3) {
        gnc_cognitive_detect_emergent_insights();
    }
}

void gnc_cognitive_amplify_pattern(const GncCognitivePattern& pattern)
{
    // Amplify cognitive pattern across distributed modules
    g_debug("Amplifying cognitive pattern from module: %s (strength: %.2f)",
            gnc_cognitive_module_name(pattern.trigger_module),
            pattern.strength);
    
    // Trigger cross-module resonance
    gnc_cognitive_broadcast_message(pattern.trigger_module,
                                   GNC_MSG_PATTERN_RESONANCE,
                                   const_cast<GncCognitivePattern*>(&pattern));
}

void gnc_cognitive_detect_emergent_insights(void)
{
    if (!g_comm_hub) {
        return;
    }
    
    g_message("Detecting emergent cognitive insights from %zu active patterns",
              g_comm_hub->active_patterns.size());
    
    // Analyze pattern interactions for emergent insights
    for (const auto& pattern : g_comm_hub->active_patterns) {
        if (pattern.strength > 0.6) {
            // Strong pattern indicates potential insight
            gnc_cognitive_generate_insight(pattern);
        }
    }
}

void gnc_cognitive_generate_insight(const GncCognitivePattern& pattern)
{
    g_message("Generating cognitive insight from pattern (module: %s, strength: %.2f)",
              gnc_cognitive_module_name(pattern.trigger_module),
              pattern.strength);
    
    // Create insight representation in Scheme
    gchar* insight_scheme = g_strdup_printf(
        "(emergent-cognitive-insight 'pattern-%s %.2f)",
        gnc_cognitive_module_name(pattern.trigger_module),
        pattern.strength);
    
    // Evaluate insight in cognitive context
    gchar* result = gnc_cognitive_scheme_eval(insight_scheme);
    if (result) {
        g_debug("Generated insight: %s", result);
        g_free(result);
    }
    
    g_free(insight_scheme);
}

const gchar* gnc_cognitive_module_name(GncCognitiveModule module)
{
    switch (module) {
        case GNC_MODULE_ATOMSPACE: return "AtomSpace";
        case GNC_MODULE_PLN: return "PLN";
        case GNC_MODULE_ECAN: return "ECAN";
        case GNC_MODULE_MOSES: return "MOSES";
        case GNC_MODULE_URE: return "URE";
        case GNC_MODULE_COGSERVER: return "CogServer";
        case GNC_MODULE_SCHEME: return "Scheme";
        default: return "Unknown";
    }
}

void gnc_cognitive_optimize_attention_flow(void)
{
    if (!g_comm_hub) {
        return;
    }
    
    g_message("Optimizing cognitive attention flow across modules");
    
    // Analyze message patterns for attention optimization
    gdouble total_activity = 0.0;
    
    for (const auto& [module, queue] : g_comm_hub->module_queues) {
        gdouble module_activity = static_cast<gdouble>(queue.size());
        total_activity += module_activity;
        
        if (module_activity > 0) {
            g_debug("Module %s activity: %.1f messages",
                    gnc_cognitive_module_name(module), module_activity);
        }
    }
    
    // Trigger adaptive attention allocation based on activity
    if (total_activity > 10.0) {
        gnc_cognitive_broadcast_message(GNC_MODULE_ECAN,
                                       GNC_MSG_ATTENTION_REALLOCATION,
                                       &total_activity);
    }
}

void gnc_cognitive_synchronize_modules(void)
{
    if (!g_comm_hub) {
        return;
    }
    
    g_message("Synchronizing cognitive modules for distributed cognition");
    
    // Send synchronization signals to all active modules
    gint64 sync_timestamp = g_get_real_time();
    
    for (const auto& [module, active] : g_comm_hub->module_status) {
        if (active) {
            gnc_cognitive_send_message(GNC_MODULE_ATOMSPACE, module,
                                     GNC_MSG_SYNCHRONIZATION,
                                     &sync_timestamp);
        }
    }
    
    // Process any pending emergent patterns
    gnc_cognitive_process_emergent_patterns();
}